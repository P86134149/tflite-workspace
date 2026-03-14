# tflite_yolox_inference.py
# 適用於 UINT8 輸入 / Float32 輸出的 TFLite 模型推論 (已硬編碼路徑)

import numpy as np
import os
import argparse
import glob
import time
from PIL import Image, ImageDraw

# 嘗試引入 TFLite Interpreter
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Error: 請安裝 tflite-runtime 或 tensorflow (pip install tensorflow)")
        exit(1)

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
        # 假設模型輸入是 256x256，將正規化座標(0-256)轉回原圖尺寸
        scale_x = w_img / 256.0; scale_y = h_img / 256.0

        # 畫 GT (綠色)
        for box in gt_boxes:
            cls_id = int(box[0])
            x1 = box[1] * scale_x; y1 = box[2] * scale_y
            x2 = box[3] * scale_x; y2 = box[4] * scale_y
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)
            draw.text((x1, y1+5), f"GT:{cls_id}", fill="#00FF00")

        # 畫 Prediction (紅色)
        for i, box in enumerate(pred_boxes):
            score = pred_scores[i]
            cls_id = int(pred_classes[i])
            if score < CONFIDENCE_THRESH: continue 
            x1 = box[0] * scale_x; y1 = box[1] * scale_y
            x2 = box[2] * scale_x; y2 = box[3] * scale_y
            draw.rectangle([x1, y1, x2, y2], outline="#FF0000", width=2)
            draw.text((x1, y1+30), f"P{cls_id}:{score:.2f}", fill="#FF0000")

        img_pil.save(output_filename)
        print(f"    [視覺化] 圖片已儲存: {output_filename}")
    except Exception as e: print(f"存圖失敗: {e}")
def save_prediction_txt(output_path, pred_boxes, pred_classes, pred_scores):
    """
    將預測結果保存為 YOLO 格式 txt: [class_id score cx cy w h] (正規化 0~1)
    使用 'w' 模式開啟，確保覆蓋舊檔，不會造成內容重複堆疊。
    """
    # 確保目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'w') as f: # 'w' 模式：每次寫入前會清空檔案
            # 如果沒有預測框，這裡就會產生一個空檔案 (符合預期)
            for i, box in enumerate(pred_boxes):
                score = pred_scores[i]
                if score < CONFIDENCE_THRESH: continue
                
                cls_id = int(pred_classes[i])
                x1, y1, x2, y2 = box
                
                # 轉換為 YOLO 格式 (Center X, Center Y, W, H) 並正規化 (0~1)
                # 假設模型輸入尺寸為 256x256
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                
                # 正規化
                ncx = cx / 256.0
                ncy = cy / 256.0
                nw = w / 256.0
                nh = h / 256.0
                
                # 寫入一行: class score cx cy w h
                f.write(f"{cls_id} {score:.6f} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")
                
        # print(f"    [Log] 預測結果已儲存: {output_filename}")
        
    except Exception as e:
        print(f"寫入 TXT 失敗: {e}")
# ----------------------------------------------------
# 3. 數學與解碼
# ----------------------------------------------------
def sigmoid(data):
    data = np.clip(data, -80, 80) 
    return 1.0 / (1.0 + np.exp(-data))

def _decode_yolo_predictions(feats: np.ndarray, stride: int, conf_thresh: float) -> tuple:

    # 如果有 batch 維度 (1, H, W, C)，移除它 改成 (H, W, C)
    if len(feats.shape) == 4:
        feats = np.squeeze(feats, axis=0)

    grid_h, grid_w, _ = feats.shape
    img_size = grid_h * stride 
    grid_y, grid_x = np.mgrid[0:grid_h, 0:grid_w]
    grid = np.stack((grid_x, grid_y), axis=-1).astype(np.float32)
    feats = feats.reshape(-1, TOTAL_CHANNELS)
    grid = grid.reshape(-1, 2)
    
    box_xy_raw = feats[:, 0:2] 
    box_wh_raw = feats[:, 2:4] 
    obj_conf_raw = feats[:, 4]
    class_scores_raw = feats[:, 5:]

    box_xy_raw[:, 0] += CENTER_BIAS_LOGIT 
    box_xy = (sigmoid(box_xy_raw) + grid) * stride
    obj_conf = sigmoid(obj_conf_raw)
    box_wh = np.exp(np.clip(box_wh_raw, -LOGIT_CLIP, LOGIT_CLIP)) * img_size
    box_wh[:, 0] *= CORRECTION_FACTOR_W 
    box_wh[:, 1] *= CORRECTION_FACTOR_H

    class_logits_clipped = np.clip(class_scores_raw, -LOGIT_CLIP, LOGIT_CLIP)
    e_x = np.exp(class_logits_clipped - np.max(class_logits_clipped, axis=-1, keepdims=True))
    class_scores = e_x / (np.sum(e_x, axis=-1, keepdims=True) + EPSILON)

    final_scores = obj_conf[:, np.newaxis] * class_scores
    max_scores = np.max(final_scores, axis=1)
    valid_mask = max_scores >= conf_thresh
    
    final_scores = final_scores[valid_mask]
    box_xy = box_xy[valid_mask]
    box_wh = box_wh[valid_mask]
    
    top_left = box_xy - box_wh / 2
    bottom_right = box_xy + box_wh / 2
    boxes = np.concatenate([top_left, bottom_right], axis=1)
    boxes = np.clip(boxes, 0, 256)
    return boxes, final_scores

def _non_max_suppression(boxes, scores, iou_thresh):
    if len(boxes) == 0: return np.array([], dtype=np.int32)
    
    if scores.ndim > 1: order = np.max(scores, axis=1).argsort()[::-1]
    else: order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # 向量化 IoU 計算
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_order = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        
        union = area_i + area_order - inter
        iou = inter / (union + EPSILON)
        
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

# ----------------------------------------------------
# 4. 評估指標
# ----------------------------------------------------
def compute_iou(box1, box2):
    xx1 = max(box1[0], box2[0]); yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2]); yy2 = min(box1[3], box2[3])
    w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
    inter = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + EPSILON)

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

def evaluate_single_image(pred_boxes, pred_classes, pred_scores, gt_boxes, iou_thresh=0.5):
    tp_count = 0; fp_count = 0; matched_gt = [False] * len(gt_boxes)
    sorted_indices = np.argsort(pred_scores)[::-1]
    results_log = []

    for idx in sorted_indices:
        p_box = pred_boxes[idx]; p_cls = pred_classes[idx]; p_score = pred_scores[idx]
        best_iou = 0.0; best_gt_idx = -1
        
        for i, gt in enumerate(gt_boxes):
            gt_cls = gt[0]; gt_box = gt[1:]
            if p_cls == gt_cls: 
                iou = compute_iou(p_box, gt_box)
                if iou > best_iou: best_iou = iou; best_gt_idx = i
        
        status = "FP"
        if best_gt_idx >= 0 and best_iou >= iou_thresh and not matched_gt[best_gt_idx]:
            tp_count += 1; matched_gt[best_gt_idx] = True; status = "TP"
        else:
            fp_count += 1
            
        results_log.append((p_cls, p_score, status, best_iou))
        
    return tp_count, fp_count, len(gt_boxes), results_log

def load_yolo_label(label_path):
    if not os.path.exists(label_path): return []
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) == 5:
                cls, cx, cy, w, h = parts
                cx *= 256; cy *= 256; w *= 256; h *= 256
                boxes.append([cls, cx-w/2, cy-h/2, cx+w/2, cy+h/2])
    return np.array(boxes)

# ----------------------------------------------------
# 5. TFLite Inference Engine
# ----------------------------------------------------
class YoloXInference:
    def __init__(self, model_file):
        if not os.path.exists(model_file): raise FileNotFoundError(f"Model not found: {model_file}")
        print(f"Loading TFLite model: {model_file}")
        
        self.interpreter = tflite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_index = self.input_details[0]['index']
        # 確保讀取到正確的輸入尺寸 (通常是 1, 256, 256, 1)
        self.nn_h = self.input_details[0]['shape'][1]
        self.nn_w = self.input_details[0]['shape'][2]
        
        # 建立 Output Index 與 Stride 的映射
        self.output_map = {}
        for i, detail in enumerate(self.output_details):
            shape = detail['shape']
            if len(shape) == 4 and shape[3] == TOTAL_CHANNELS:
                h_grid = shape[1]
                stride = int(self.nn_h / h_grid)
                self.output_map[i] = stride
                print(f"  Mapped Output Index {i}: Shape {shape} -> Stride {stride}")

        self.t_pre = 0.0; self.t_infer = 0.0; self.t_post = 0.0

    def preprocess_image(self, image_path):
        if not os.path.exists(image_path): raise FileNotFoundError(f"Image not found: {image_path}")
        
        # 1. 讀取圖片並轉為灰階 (L mode: 0-255)
        img = Image.open(image_path).convert('L')
        
        t0 = time.perf_counter()
        
        # 2. Resize
        img = img.resize((self.nn_w, self.nn_h))
        
        # 3. 修正點：直接轉為 UINT8 (0~255)，不要減去 128
        input_data = np.asarray(img, dtype=np.uint8)
        
        # 4. 增加維度 (Batch, H, W, Channel)
        # 這裡假設模型輸入需要 4 維 (1, 256, 256, 1)
        final_input = np.expand_dims(np.expand_dims(input_data, axis=-1), axis=0)
        
        self.t_pre = (time.perf_counter() - t0) * 1000
        return final_input

    def launch_inference(self, input_data):
        t0 = time.perf_counter()
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        self.t_infer = (time.perf_counter() - t0) * 1000

    def get_and_process_outputs(self):
        t0 = time.perf_counter()
        all_detections = []
        
        for idx, stride in self.output_map.items():
            # 直接獲取 F32 輸出
            output_tensor = self.interpreter.get_tensor(self.output_details[idx]['index'])
            
            boxes, scores = _decode_yolo_predictions(output_tensor, stride, CONFIDENCE_THRESH)
            
            if boxes.size > 0:
                max_scores = np.max(scores, axis=1)
                max_classes = np.argmax(scores, axis=1)
                dets = np.hstack([boxes, max_scores[:, np.newaxis], max_classes[:, np.newaxis]])
                all_detections.append(dets)

        if not all_detections: 
            self.t_post = (time.perf_counter() - t0) * 1000
            return np.array([])

        final_detections = np.concatenate(all_detections, axis=0)
        keep_idx = _non_max_suppression(final_detections[:, :4], final_detections[:, 4], NMS_THRESH)
        if len(keep_idx) > MAX_DETECTIONS: keep_idx = keep_idx[:MAX_DETECTIONS]
        
        self.t_post = (time.perf_counter() - t0) * 1000
        return final_detections[keep_idx]

# ----------------------------------------------------
# 6. Main
# ----------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 預設路徑 (請根據需求修改)
    DEFAULT_MODEL = r"D:\Downloads\stm32MPU\Models\ST_YOLO_X\9class-model\9class_fold5_yoloX_PerTensor_quant_uint8_float32_random_1.tflite"
    DEFAULT_SOURCE = r"D:\Downloads\YOLO_Database\mit-bih_code\9_class\fold5\val"

    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Path to .tflite model")
    parser.add_argument("-s", "--source", default=DEFAULT_SOURCE, help="Path to image directory")
    args = parser.parse_args()
    
    # OUTPUT_DIR = "tflite_inference_results/0116_FOLD2"
    # LABEL_DIR = os.path.join(OUTPUT_DIR, "labels") # 專門放 txt 的資料夾
    # if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    # if not os.path.exists(LABEL_DIR): os.makedirs(LABEL_DIR)

    try:
        engine = YoloXInference(args.model)
        if os.path.isdir(args.source):
            img_files = glob.glob(os.path.join(args.source, "*.jpg")); is_batch = True
        else:
            img_files = [args.source]; is_batch = False

        # 初始化統計變數
        tp=0; fp=0; gt=0 
        sum_pre = 0.0; sum_infer = 0.0; sum_post = 0.0
        
        all_pred_stats = []; all_gt_counts = {}

        print(f"--- 處理 {len(img_files)} 張圖片 ---")
        print(f"    Source: {args.source}")
        
        for img_path in img_files:
            bn = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(args.source if is_batch else os.path.dirname(img_path), bn + ".txt")

            # 1. 推論流程
            data = engine.preprocess_image(img_path)
            engine.launch_inference(data)
            preds = engine.get_and_process_outputs()
            
            # 2. 累加時間 (Latency Accumulation)
            sum_pre += engine.t_pre
            sum_infer += engine.t_infer
            sum_post += engine.t_post
            
            if len(preds) > 0:
                cx = (preds[:,0]+preds[:,2])/2
                preds = preds[np.argsort(cx)]
            gts = load_yolo_label(txt_path)

            ## 🚨 TXT 儲存
            # pred_txt_path = os.path.join(LABEL_DIR, f"{bn}.txt")

            # if len(preds) > 0:
            #     save_prediction_txt(pred_txt_path, 
            #                       preds[:, :4], # Boxes
            #                       preds[:, 5],  # Classes
            #                       preds[:, 4])  # Scores
            # else:
            #     # 如果沒有偵測到任何物體，也要產生一個空檔案覆蓋舊檔
            #     save_prediction_txt(pred_txt_path, [], [], [])
            
            # # 儲存圖片 (可選)
            # save_path = os.path.join(OUTPUT_DIR, f"pred_{bn}.jpg")
            # save_debug_image(img_path, gts, preds[:, :4] if len(preds)>0 else [], 
            #                  preds[:, 4] if len(preds)>0 else [], 
            #                  preds[:, 5] if len(preds)>0 else [], save_path)
            
            
            for g in gts:
                cls_id = int(g[0])
                all_gt_counts[cls_id] = all_gt_counts.get(cls_id, 0) + 1

            n_gt = len(gts)
            if len(gts) > 0:
                t, f, n, logs = evaluate_single_image(preds[:,:4] if len(preds)>0 else [], 
                                                      preds[:,5] if len(preds)>0 else [], 
                                                      preds[:,4] if len(preds)>0 else [], gts, EVAL_IOU_THRESH)
                tp+=t; fp+=f; gt+=n
                for log in logs:
                    is_tp = 1 if log[2] == "TP" else 0
                    all_pred_stats.append((int(log[0]), log[1], is_tp))
            else:
                logs = [(preds[i,5], preds[i,4], "Pred", 0.0) for i in range(len(preds))]
                for i in range(len(preds)):
                    all_pred_stats.append((int(preds[i,5]), preds[i,4], 0))

            if not is_batch or img_path == img_files[0]:
                print(f"\n--- {bn} ---")
                print(f"    [Time] Infer: {engine.t_infer:.4f}ms") # 單張顯示只顯示 Infer
                print(f"    GT: {n_gt}, Pred: {len(preds)}")
                for l in logs[:5]: print(f"      C{int(l[0])} {l[1]:.2f} -> {l[2]} IoU={l[3]:.2f}")

        # 3. 最終結果與 mAP 計算
        if gt > 0:
            p = tp/(tp+fp+1e-6); r = tp/(gt+1e-6); f1 = 2*p*r/(p+r+1e-6)
            aps = []
            unique_classes = sorted(all_gt_counts.keys())
            for cls_id in unique_classes:
                n_gt_cls = all_gt_counts[cls_id]
                cls_preds = [item for item in all_pred_stats if item[0] == cls_id]
                if n_gt_cls == 0 and len(cls_preds) == 0: continue
                if len(cls_preds) == 0: aps.append(0); continue
                cls_preds.sort(key=lambda x: x[1], reverse=True)
                tp_list = np.array([x[2] for x in cls_preds])
                fp_list = 1 - tp_list
                tp_cumsum = np.cumsum(tp_list); fp_cumsum = np.cumsum(fp_list)
                recalls = tp_cumsum / (n_gt_cls + 1e-16)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
                ap = compute_ap(recalls, precisions)
                aps.append(ap)
            
            mAP = np.mean(aps) if aps else 0.0

            print(f"\n=== TFLite Result (IoU>={EVAL_IOU_THRESH}, Conf>={CONFIDENCE_THRESH}) ===")
            print(f"Precision: {p:.4f}")
            print(f"Recall:    {r:.4f}")
            print(f"mAP@0.5:   {mAP:.4f}") 
            print(f"F1-Score:  {f1:.4f}")
            
            # 4. 🚨 新增：Latency Analysis 打印區塊 🚨
            n = len(img_files)
            print("\n=== Latency Analysis ===")
            print(f"Avg Pre-process:  {sum_pre/n:.4f} ms")
            print(f"Avg Inference:    {sum_infer/n:.4f} ms (CPU)") # TFLite 通常在 CPU 跑
            print(f"Avg Post-process: {sum_post/n:.4f} ms (Decode+NMS)")
            
            avg_total = (sum_pre + sum_infer + sum_post) / n
            print(f"Avg Total E2E:    {avg_total:.4f} ms")
            
            avg_fps = 1000.0 / avg_total if avg_total > 0 else 0
            print(f"System FPS:       {avg_fps:.2f} fps")
            print("========================================")

    except Exception as e:
        import traceback; traceback.print_exc()

#python3 st_yoloX_inference_TFLITE.py -m fold5_1105_best_PerTensor_quant_uint8_float32_random_1.tflite -s fold5