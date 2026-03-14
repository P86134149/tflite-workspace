from stai_mpu import stai_mpu_network
import numpy as np
import os
import argparse
import glob
import time
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# 1. 全局配置 (與您提供的 YOLO_LC 參數一致)
# ==============================================================================
ANCHORS = [
    [0.076023, 0.258508],
    [0.163031, 0.413531],
    [0.234769, 0.702585],
    [0.427054, 0.715892],
    [0.748154, 0.857092]
]

NUM_CLASSES = 5 
TOTAL_CHANNELS = 50 
CONFIDENCE_THRESH = 0.5 
NMS_THRESH = 0.5         
MAX_DETECTIONS = 50       
LOGIT_CLIP = 10.0         
EPSILON = 1e-6            
EVAL_IOU_THRESH = 0.5 

# ==============================================================================
# 2. 核心工具函式
# ==============================================================================
def sigmoid(x):
    x = np.clip(x, -80, 80)
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-6)

def compute_iou_vectorized(box, boxes_array):
    """
    計算一個 box 與多個 boxes 的 IoU向量化
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
    xx1, yy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xx2, yy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    w, h = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
    inter = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

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
def _non_max_suppression(boxes, scores, iou_thresh):
    if len(boxes) == 0: return np.array([], dtype=np.int32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    order = scores.argsort()[::-1]
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

def save_debug_image(image_path, gt_boxes, pred_boxes, pred_scores, pred_classes, output_filename):
    if not os.path.exists(image_path): return
    try:
        img_pil = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img_pil)
        w_img, h_img = img_pil.size
        scale_x = w_img / 256.0; scale_y = h_img / 256.0
        for box in gt_boxes:
            draw.rectangle([box[1]*scale_x, box[2]*scale_y, box[3]*scale_x, box[4]*scale_y], outline="#00FF00", width=2)
        for i, box in enumerate(pred_boxes):
            if pred_scores[i] < CONFIDENCE_THRESH: continue
            draw.rectangle([box[0]*scale_x, box[1]*scale_y, box[2]*scale_x, box[3]*scale_y], outline="#FF0000", width=2)
            draw.text((box[0]*scale_x, box[1]*scale_y), f"P{int(pred_classes[i])}:{pred_scores[i]:.2f}", fill="#FF0000")
        img_pil.save(output_filename)
    except Exception as e: print(f"Viz Error: {e}")

# ----------------------------------------------------
# 3. 新推論引擎 (YoloLCInference)
# ----------------------------------------------------
class YoloLCInference:
    def __init__(self, model_file):
        if not os.path.exists(model_file): raise FileNotFoundError(f"NBG Model not found: {model_file}")
        print(f"Loading NBG model (YOLO LC): {model_file}")
        self.stai_mpu_model = stai_mpu_network(model_path=model_file, use_hw_acceleration=True)
        self.output_infos = self.stai_mpu_model.get_output_infos()
        self.nn_h = 256; self.nn_w = 256
        self.t_pre = 0.0; self.t_infer = 0.0; self.t_post = 0.0

    def preprocess_image(self, image_path):
        # 照抄 YOLOX uint8 處理方式
        img = Image.open(image_path).convert('L')
        t0 = time.perf_counter()
        img = img.resize((self.nn_w, self.nn_h))
        input_data = np.asarray(img, dtype=np.uint8)
        final_input = np.expand_dims(np.expand_dims(input_data, axis=-1), axis=0)
        self.t_pre = (time.perf_counter() - t0) * 1000
        return final_input

    def launch_inference(self, input_data):
        t0 = time.perf_counter()
        self.stai_mpu_model.set_input(0, input_data)
        self.stai_mpu_model.run()
        self.t_infer = (time.perf_counter() - t0) * 1000

    def get_and_process_outputs(self):
        t0 = time.perf_counter()
        # 假設 YOLO LC 只有一個主要輸出層 (例如 16x16)
        output_tensor = self.stai_mpu_model.get_output(index=0)
        shape = self.output_infos[0].get_shape()
        target_shape = tuple([s for s in shape if s > 1]) # 通常是 (16, 16, 50)
        
        raw_data = np.frombuffer(output_tensor, dtype=np.float32)
        feats = raw_data.reshape(target_shape)
        
        # 1. 整理 Shape
        features_per_anchor = 5 + NUM_CLASSES 
        num_anchors = feats.shape[-1] // features_per_anchor
        grid_h, grid_w = feats.shape[:2]
        feats = feats.reshape(grid_h, grid_w, num_anchors, features_per_anchor)
        
        # 2. 準備 Grid 與 Anchors
        grid_y, grid_x = np.mgrid[0:grid_h, 0:grid_w]
        grid_xy = np.stack((grid_x, grid_y), axis=-1).astype(np.float32)
        grid_xy = np.expand_dims(grid_xy, axis=2) 
        anchors_tensor = np.array(ANCHORS[:num_anchors], dtype=np.float32).reshape(1, 1, num_anchors, 2)

        # 3. 核心解碼
        bx_by = sigmoid(feats[..., 0:2])
        bw_bh = (np.exp(np.clip(feats[..., 2:4], -5.0, 5.0)) * anchors_tensor) / float(grid_w)
        pred_xy = (bx_by + grid_xy) / float(grid_w)
        
        # 4. 取得分數
        obj_conf = sigmoid(feats[..., 4:5])
        cls_scores = softmax(feats[..., 5:])
        final_scores_map = (obj_conf * cls_scores).reshape(-1, NUM_CLASSES)
        
        # 5. 座標轉換 (x1, y1, x2, y2)
        pred_xy = pred_xy.reshape(-1, 2)
        pred_wh = bw_bh.reshape(-1, 2)
        x1 = (pred_xy[:, 0] - pred_wh[:, 0] / 2) * 256.0
        y1 = (pred_xy[:, 1] - pred_wh[:, 1] / 2) * 256.0
        x2 = (pred_xy[:, 0] + pred_wh[:, 0] / 2) * 256.0
        y2 = (pred_xy[:, 1] + pred_wh[:, 1] / 2) * 256.0
        boxes = np.stack([x1, y1, x2, y2], axis=-1)
        
        # 6. NMS (使用 NumPy 取代 CV2)
        max_scores = np.max(final_scores_map, axis=1)
        keep_mask = max_scores >= CONFIDENCE_THRESH
        
        boxes = boxes[keep_mask]
        scores = max_scores[keep_mask]
        classes = np.argmax(final_scores_map[keep_mask], axis=1)

        if len(boxes) == 0:
            self.t_post = (time.perf_counter() - t0) * 1000
            return np.array([])

        keep_idx = _non_max_suppression(boxes, scores, NMS_THRESH)
        
        # 為了與後續 mAP 邏輯相容，合併成 [x1, y1, x2, y2, score, class]
        result = np.hstack([
            boxes[keep_idx], 
            scores[keep_idx, np.newaxis], 
            classes[keep_idx, np.newaxis]
        ])
        
        self.t_post = (time.perf_counter() - t0) * 1000
        return result

# ==============================================================================
# 5. 主流程
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True); parser.add_argument("-s", "--source", required=True)
    args = parser.parse_args()
    
    OUTPUT_DIR = "lc_v1_nbg_results"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    try:
        engine = YoloLCInference(args.model)
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
            save_path = os.path.join(OUTPUT_DIR, f"pred_{bn}.jpg")
            # save_debug_image(img_path, gts, preds[:, :4], preds[:, 4], preds[:, 5], save_path)
            
            # Save Prediction TXT
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
