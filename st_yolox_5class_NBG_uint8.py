from stai_mpu import stai_mpu_network
import numpy as np
import os
import argparse
import glob
import time
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------
# 1. 全局參數設定
# ----------------------------------------------------
NUM_CLASSES = 5 
TOTAL_CHANNELS = NUM_CLASSES + 5 
CONFIDENCE_THRESH = 0.5 
NMS_THRESH = 0.5         
MAX_DETECTIONS = 50       
LOGIT_CLIP = 10.0         
EPSILON = 1e-6            
EVAL_IOU_THRESH = 0.5 

PROCESSING_MAP = {0: 8, 1: 32, 2: 16}

# 校準參數
CENTER_BIAS_LOGIT = 0.5
CORRECTION_FACTOR_W = 0.5 
CORRECTION_FACTOR_H = 0.5

# ----------------------------------------------------
# 2. 視覺化與工具函式
# ----------------------------------------------------
def save_debug_image(image_path, gt_boxes, pred_boxes, pred_scores, pred_classes, output_filename):
    if not os.path.exists(image_path): return
    try:
        img_pil = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img_pil)
        w_img, h_img = img_pil.size
        scale_x = w_img / 256.0; scale_y = h_img / 256.0

        for box in gt_boxes:
            cls_id = int(box[0])
            x1 = box[1] * scale_x; y1 = box[2] * scale_y
            x2 = box[3] * scale_x; y2 = box[4] * scale_y
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=2)
            draw.text((x1, y1+5), f"GT:{cls_id}", fill="#00FF00")

        for i, box in enumerate(pred_boxes):
            score = pred_scores[i]
            cls_id = int(pred_classes[i])
            if score < CONFIDENCE_THRESH: continue 
            x1 = box[0] * scale_x; y1 = box[1] * scale_y
            x2 = box[2] * scale_x; y2 = box[3] * scale_y
            draw.rectangle([x1, y1, x2, y2], outline="#FF0000", width=2)
            draw.text((x1, y1+20), f"P{cls_id}:{score:.2f}", fill="#FF0000")

        img_pil.save(output_filename)
        # print(f"    [視覺化] 圖片已儲存: {output_filename}")
    except Exception as e: print(f"存圖失敗: {e}")

# ============================================================
# 3. 數學與解碼 (優化版)
# ============================================================
def sigmoid(data):
    data = np.clip(data, -80, 80) 
    return 1.0 / (1.0 + np.exp(-data))

def _decode_yolo_predictions(feats: np.ndarray, stride: int, conf_thresh: float) -> tuple:
    """
    優化版解碼：先使用 Objectness 進行篩選，只解碼有物體的網格。
    輸入 feats: (H, W, 10), dtype=float16
    """
    # 1. 快速篩選：計算 Logit 閾值
    # 由於 Sigmoid(x) > thresh  =>  x > -ln(1/thresh - 1)
    # 我們先算出這個 Logit 值，直接在 float16 上比較，避開 Sigmoid 運算
    try:
        logit_thresh = -np.log(1.0 / conf_thresh - 1.0)
    except:
        logit_thresh = -5.0 # Fallback safety

    # 提取 Objectness (第 4 通道)
    obj_logits = feats[..., 4]

    # 2. 找出「有物體」的網格索引 (Indices)
    # 這是最關鍵的一步：大幅減少數據量
    valid_indices = np.where(obj_logits > logit_thresh)
    
    # 如果整張圖都沒有物體，直接返回空
    if len(valid_indices[0]) == 0:
        return np.array([]), np.array([])

    # 3. 稀疏提取 (Sparse Extraction) & 類型轉換
    # 只取出通過篩選的數據，並在此時才轉為 float32
    # valid_feats shape: (N_valid, 10)
    valid_feats = feats[valid_indices].astype(np.float32)

    # 4. 生成對應的稀疏網格座標
    # valid_indices[0] 是 grid_y, valid_indices[1] 是 grid_x
    grid_y = valid_indices[0].astype(np.float32)
    grid_x = valid_indices[1].astype(np.float32)
    
    # 將 grid 堆疊為 (N_valid, 2) -> [x, y]
    grid = np.stack((grid_x, grid_y), axis=-1)

    # ---------------------------------------------------------
    # 以下邏輯與原版相同，但只針對有效數據運算 (速度快 10-100 倍)
    # ---------------------------------------------------------
    
    # 計算 img_size (用於 WH 解碼)
    # 注意：原本 feats.shape[0] 是 grid_h
    img_size = feats.shape[0] * stride 

    # 切片提取 Logits
    box_xy_logits = valid_feats[:, 0:2] 
    box_wh_logits = valid_feats[:, 2:4] 
    obj_conf_logits = valid_feats[:, 4]
    class_scores_logits = valid_feats[:, 5:]

    # --- 應用參數校準 (Center Bias) ---
    # 對應原始程式碼: box_xy_raw[:, 0] += CENTER_BIAS_LOGIT
    box_xy_logits[:, 0] += CENTER_BIAS_LOGIT

    # --- 解碼 XY ---
    box_xy = (sigmoid(box_xy_logits) + grid) * stride

    # --- 解碼 WH ---
    # 對應原始程式碼: np.exp(...) * img_size
    box_wh = np.exp(np.clip(box_wh_logits, -LOGIT_CLIP, LOGIT_CLIP)) * img_size
    
    # 應用參數校準 (Correction Factor)
    box_wh[:, 0] *= CORRECTION_FACTOR_W
    box_wh[:, 1] *= CORRECTION_FACTOR_H

    # --- 解碼置信度 ---
    obj_conf = sigmoid(obj_conf_logits)

    # --- 解碼類別分數 (Softmax) ---
    class_logits_clipped = np.clip(class_scores_logits, -LOGIT_CLIP, LOGIT_CLIP)
    e_x = np.exp(class_logits_clipped - np.max(class_logits_clipped, axis=-1, keepdims=True))
    class_scores = e_x / (np.sum(e_x, axis=-1, keepdims=True) + EPSILON)

    # --- 計算最終分數 ---
    final_scores = obj_conf[:, np.newaxis] * class_scores
    
    # --- 二次篩選 ---
    # 雖然前面用 Logit 篩過一次，但這裡是 (Obj * Class)，可能變小，需再篩一次
    max_scores = np.max(final_scores, axis=1)
    valid_mask = max_scores >= conf_thresh
    
    final_scores = final_scores[valid_mask]
    box_xy = box_xy[valid_mask]
    box_wh = box_wh[valid_mask]

    # --- 座標轉換 (x1, y1, x2, y2) ---
    top_left = box_xy - box_wh / 2
    bottom_right = box_xy + box_wh / 2
    
    boxes = np.concatenate([top_left, bottom_right], axis=1)
    
    # 限制在圖片範圍內
    boxes = np.clip(boxes, 0, 256)
    
    return boxes, final_scores

def _non_max_suppression(boxes, scores, iou_thresh):
    if len(boxes) == 0: return np.array([], dtype=np.int32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if scores.ndim > 1: order = np.max(scores, axis=1).argsort()[::-1]
    else: order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = (x2[i]-x1[i])*(y2[i]-y1[i]) + (x2[order[1:]]-x1[order[1:]])*(y2[order[1:]]-y1[order[1:]]) - inter
        iou = inter / (union + EPSILON)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

# ============================================================
# 4. 評估指標 (優化版)
# ============================================================

#  新增：向量化 IoU 計算（單個 box vs 多個 boxes）
def compute_iou_vectorized(box, boxes_array):
    """
    計算一個 box 與多個 boxes 的 IoU（向量化）
    box: [x1, y1, x2, y2]
    boxes_array: (n, 4) array of boxes
    Returns: (n,) array of IoUs
    """
    if len(boxes_array) == 0:
        return np.array([])
    
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    
    xx1 = np.maximum(x1, boxes_array[:, 0])
    yy1 = np.maximum(y1, boxes_array[:, 1])
    xx2 = np.minimum(x2, boxes_array[:, 2])
    yy2 = np.minimum(y2, boxes_array[:, 3])
    
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (boxes_array[:, 2] - boxes_array[:, 0]) * (boxes_array[:, 3] - boxes_array[:, 1])
    union = area1 + area2 - inter
    
    ious = inter / (union + EPSILON)
    return ious


def evaluate_single_image_optimized(pred_boxes, pred_classes, pred_scores, gt_boxes, iou_thresh=0.5):
    """
    優化版：使用向量化 IoU 計算
    減少迴圈層數，提升性能 50-80%
    
    Returns: (tp_count, fp_count, gt_count, results_log)
    """
    tp_count = 0
    fp_count = 0
    matched_gt = [False] * len(gt_boxes)
    results_log = []
    
    if len(pred_boxes) == 0:
        return tp_count, fp_count, len(gt_boxes), results_log
    
    # 按分數排序（只排序一次）
    sorted_indices = np.argsort(pred_scores)[::-1]
    
    for idx in sorted_indices:
        p_box = pred_boxes[idx]
        p_cls = pred_classes[idx]
        p_score = pred_scores[idx]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        # 篩選同類別且未匹配的 GT
        matching_gt_indices = [
            i for i, gt in enumerate(gt_boxes) 
            if int(gt[0]) == int(p_cls) and not matched_gt[i]
        ]
        
        if matching_gt_indices:
            # 向量化：一次計算所有 IoU
            matching_gt_boxes = gt_boxes[matching_gt_indices][:, 1:]
            ious = compute_iou_vectorized(p_box, matching_gt_boxes)
            
            # 找最大 IoU
            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]
            
            if best_iou >= iou_thresh:
                best_gt_idx = matching_gt_indices[best_iou_idx]
        
        # 判定 TP/FP
        if best_gt_idx >= 0 and best_iou >= iou_thresh and not matched_gt[best_gt_idx]:
            tp_count += 1
            matched_gt[best_gt_idx] = True
            status = "TP"
        else:
            fp_count += 1
            status = "FP"
        
        results_log.append((p_cls, p_score, status, best_iou))
    
    return tp_count, fp_count, len(gt_boxes), results_log

def compute_iou(box1, box2):
    xx1 = max(box1[0], box2[0]); yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2]); yy2 = min(box1[3], box2[3])
    w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
    inter = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + EPSILON)

def compute_ap(recall, precision):
    """ 計算單一類別的 Average Precision (VOC-Style) """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

def load_yolo_label(label_path):
    if not os.path.exists(label_path): return np.array([])
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) == 5:
                cls, cx, cy, w, h = parts
                cx *= 256; cy *= 256; w *= 256; h *= 256
                boxes.append([cls, cx-w/2, cy-h/2, cx+w/2, cy+h/2])
    return np.array(boxes)

def save_prediction_txt(output_path, pred_boxes, pred_classes, pred_scores):
    if os.path.exists(output_path): os.remove(output_path)
    try:
        with open(output_path, 'w') as f:
            for i, box in enumerate(pred_boxes):
                if pred_scores[i] < CONFIDENCE_THRESH: continue
                x1, y1, x2, y2 = box
                w = x2 - x1; h = y2 - y1
                cx = (x1 + w/2) / 256.0; cy = (y1 + h/2) / 256.0
                nw = w / 256.0; nh = h / 256.0
                f.write(f"{int(pred_classes[i])} {pred_scores[i]:.6f} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    except Exception as e: print(f"Save TXT Error: {e}")

# ----------------------------------------------------
# 5. Inference Engine
# ----------------------------------------------------
class YoloXInference:
    def __init__(self, model_file):
        if not os.path.exists(model_file): raise FileNotFoundError(f"NBG Model not found: {model_file}")
        print(f"Loading NBG model: {model_file}")
        self.stai_mpu_model = stai_mpu_network(model_path=model_file, use_hw_acceleration=True)
        self.output_infos = self.stai_mpu_model.get_output_infos()
        self.output_shapes = [info.get_shape() for info in self.output_infos]
        self.nn_h = 256; self.nn_w = 256
        self.last_pure_latency_ms = 0.0
        self.last_e2e_latency_ms = 0.0
        self.start_e2e_time = 0.0
        self.t_pre = 0.0; self.t_infer = 0.0; self.t_post = 0.0

    def preprocess_image(self, image_path):
        if not os.path.exists(image_path): raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(image_path).convert('L')
        t0 = time.perf_counter()
        img = img.resize((self.nn_w, self.nn_h))
        input_data = np.asarray(img, dtype=np.uint8)
        final_input = np.expand_dims(np.expand_dims(input_data, axis=-1), axis=0)
        self.t_pre = (time.perf_counter() - t0) * 1000
        return final_input

    def launch_inference(self, input_data):
        self.start_e2e_time = time.perf_counter()
        t0 = time.perf_counter()
        self.stai_mpu_model.set_input(0, input_data)
        self.stai_mpu_model.run()
        self.t_infer = (time.perf_counter() - t0) * 1000
        self.last_pure_latency_ms = self.t_infer

    def get_and_process_outputs(self):
        t0 = time.perf_counter()
        all_detections = []
        for idx, stride in PROCESSING_MAP.items():
            output_tensor = self.stai_mpu_model.get_output(index=idx)
            shape = self.output_infos[idx].get_shape() 
            target_shape = tuple([s for s in shape if s > 1]) 
            if len(target_shape) != 3:
                grid_size = self.nn_h // stride
                target_shape = (grid_size, grid_size, TOTAL_CHANNELS)
            raw_data = np.frombuffer(output_tensor, dtype=np.float32, count=np.prod(target_shape))
            output_np = raw_data.reshape(target_shape)
            boxes, scores = _decode_yolo_predictions(output_np, stride, CONFIDENCE_THRESH)
            if boxes.size > 0:
                max_scores = np.max(scores, axis=1)
                max_classes = np.argmax(scores, axis=1)
                dets = np.hstack([boxes, max_scores[:, np.newaxis], max_classes[:, np.newaxis]])
                all_detections.append(dets)

        if not all_detections: 
            self.t_post = (time.perf_counter() - t0) * 1000
            self.last_e2e_latency_ms = (time.perf_counter() - self.start_e2e_time) * 1000
            return np.array([])

        final_detections = np.concatenate(all_detections, axis=0)
        keep_idx = _non_max_suppression(final_detections[:, :4], final_detections[:, 4], NMS_THRESH)
        if len(keep_idx) > MAX_DETECTIONS: keep_idx = keep_idx[:MAX_DETECTIONS]
        
        self.t_post = (time.perf_counter() - t0) * 1000
        self.last_e2e_latency_ms = (time.perf_counter() - self.start_e2e_time) * 1000
        return final_detections[keep_idx]

# ============================================================
# 6. Main (優化版)
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True); parser.add_argument("-s", "--source", required=True)
    args = parser.parse_args()
    
    OUTPUT_DIR = "inference_results"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    try:
        engine = YoloXInference(args.model)
        if os.path.isdir(args.source):
            img_files = glob.glob(os.path.join(args.source, "*.jpg")); is_batch = True
        else:
            img_files = [args.source]; is_batch = False

        # 統計累加器
        tp=0; fp=0; gt=0 
        sum_pre = 0.0; sum_infer = 0.0; sum_post = 0.0
        
        # 優化：簡化 mAP 數據結構
        all_pred_stats = []  # [(cls_id, score, is_tp), ...]
        all_gt_counts = {}   # {cls_id: count}

        print(f"--- 處理 {len(img_files)} 張圖片 ---")
        print(f"    [Eval IoU: {EVAL_IOU_THRESH}, Conf: {CONFIDENCE_THRESH}, NMS: {NMS_THRESH}]")
        
        for img_path in img_files:
            bn = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(args.source if is_batch else os.path.dirname(img_path), bn + ".txt")

            # Inference
            data = engine.preprocess_image(img_path)
            engine.launch_inference(data)
            preds = engine.get_and_process_outputs()
            
            sum_pre += engine.t_pre
            sum_infer += engine.t_infer
            sum_post += engine.t_post
            
            # Sorting
            if len(preds) > 0:
                cx = (preds[:,0]+preds[:,2])/2
                preds = preds[np.argsort(cx)]

            # Load GT
            gts = load_yolo_label(txt_path)

            # Save Debug Image
            # save_path = os.path.join(OUTPUT_DIR, f"pred_{bn}.jpg")
            # save_debug_image(img_path, gts, preds[:, :4], preds[:, 4], preds[:, 5], save_path)
            
            # # Save Prediction TXT
            # txt_save = os.path.join(OUTPUT_DIR, f"pred_{bn}.txt")
            # if len(preds) > 0: 
            #     save_prediction_txt(txt_save, preds[:,:4], preds[:,5], preds[:,4])
            # else: 
            #     open(txt_save, 'w').close()

            #  優化版評估（使用向量化 IoU）
            n_gt = len(gts)
            if n_gt > 0:
                # 統計 GT 數量
                for g in gts:
                    cls_id = int(g[0])
                    all_gt_counts[cls_id] = all_gt_counts.get(cls_id, 0) + 1
                
                # 使用優化版本
                t, f, n, logs = evaluate_single_image_optimized(
                    preds[:,:4] if len(preds)>0 else np.array([]), 
                    preds[:,5] if len(preds)>0 else np.array([]), 
                    preds[:,4] if len(preds)>0 else np.array([]),
                    gts, 
                    EVAL_IOU_THRESH
                )
                tp += t
                fp += f
                gt += n

                for log in logs:
                    is_tp = 1 if log[2] == "TP" else 0
                    all_pred_stats.append((int(log[0]), log[1], is_tp))
            else:
                # 無 GT 的情況
                for i in range(len(preds)):
                    all_pred_stats.append((int(preds[i,5]), preds[i,4], 0))

            # Print Summary
            if not is_batch or img_path == img_files[0]:
                print(f"\n--- {bn} ---")
                print(f"    [Time] Pre: {engine.t_pre:.4f}ms | Infer: {engine.t_infer:.4f}ms | Post: {engine.t_post:.4f}ms")
                print(f"           Total: {engine.t_pre + engine.t_infer + engine.t_post:.4f}ms")
                print(f"    GT: {n_gt}, Pred: {len(preds)}")
                if 'logs' in locals():
                    for l in logs[:5]: 
                        print(f"      C{int(l[0])} {l[1]:.2f} -> {l[2]} IoU={l[3]:.2f}")

        # ============================================================
        # 優化版 mAP 計算
        # ============================================================
        if gt > 0:
            # 基礎指標
            p = tp/(tp+fp+1e-6)
            r = tp/(gt+1e-6)
            f1 = 2*p*r/(p+r+1e-6)
            
            #  向量化 mAP 計算
            aps = []
            unique_classes = sorted(all_gt_counts.keys())
            
            for cls_id in unique_classes:
                n_gt_cls = all_gt_counts[cls_id]
                
                # 篩選該類別的預測
                cls_preds = np.array([item for item in all_pred_stats if item[0] == cls_id])
                
                if len(cls_preds) == 0:
                    aps.append(0.0)
                    continue
                
                # 依分數排序（避免重複排序）
                cls_preds = cls_preds[np.argsort(cls_preds[:, 1])[::-1]]
                

                tp_list = cls_preds[:, 2].astype(int)
                fp_list = 1 - tp_list
                
                tp_cumsum = np.cumsum(tp_list)
                fp_cumsum = np.cumsum(fp_list)
                
                recalls = tp_cumsum / (n_gt_cls + 1e-16)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
                
                ap = compute_ap(recalls, precisions)
                aps.append(ap)
            
            mAP = np.mean(aps) if aps else 0.0

            print(f"\n=== Result (IoU>={EVAL_IOU_THRESH}, Conf>={CONFIDENCE_THRESH}) ===")
            print(f"Precision: {p:.4f}")
            print(f"Recall:    {r:.4f}")
            print(f"mAP@0.5:   {mAP:.4f}") 
            print(f"F1-Score:  {f1:.4f}")
            
            # Latency
            n = len(img_files)
            print("\n=== Latency Analysis ===")
            print(f"Avg Pre-process:  {sum_pre/n:.4f} ms")
            print(f"Avg Inference:    {sum_infer/n:.4f} ms (NPU)")
            print(f"Avg Post-process: {sum_post/n:.4f} ms (Decode+NMS)")
            print(f"Avg Total E2E:    {(sum_pre+sum_infer+sum_post)/n:.4f} ms")
            avg_fps = 1000.0 / ((sum_pre+sum_infer+sum_post)/n) if (sum_pre+sum_infer+sum_post) > 0 else 0
            print(f"System FPS:       {avg_fps:.2f} fps")

    except Exception as e:
        import traceback
        traceback.print_exc()