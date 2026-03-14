import os
import argparse
import glob
import time
import numpy as np
from PIL import Image, ImageDraw

# 嘗試引入 TFLite Interpreter
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Error: 請安裝 tflite-runtime 或 tensorflow")
        exit(1)

# ==============================================================================
# 1. 全局配置 (Configuration)
# ==============================================================================
# 請務必確認這裡的 Anchor 設定與 user_config.yaml 訓練時完全一致
# 根據您的最新指示，使用單一 Anchor
ANCHORS = [
    [0.076023, 0.258508],
    [0.163031, 0.413531],
    [0.234769, 0.702585],
    [0.427054, 0.715892],
    [0.748154, 0.857092]
]
# ANCHORS = [
#   [0.126023, 0.758508],
#   [0.142222, 0.883531],
#   [0.143031, 0.583531],
#   [0.204769, 0.602585],
#   [0.149922, 0.962779]
# ]
NUM_CLASSES = 5
CONFIDENCE_THRESH = 0.5  # 信心度閾值
NMS_THRESH = 0.5        # NMS 閾值
EVAL_IOU_THRESH = 0.5    # 計算 TP 用的 IOU 閾值

EPSILON = 1e-6
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

def _non_max_suppression(boxes, scores, iou_thresh):
    if len(boxes) == 0: return np.array([], dtype=np.int32)
    
    # 確保 boxes 是 numpy array
    boxes = np.array(boxes)
    scores = np.array(scores)

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

def compute_iou(box1, box2):
    """ 計算 IoU: box = [x1, y1, x2, y2] """
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

def compute_ap(recall, precision):
    """ 計算 Average Precision """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

# ==============================================================================
# 3. 使用者指定的存圖/存檔函式
# ==============================================================================
def save_debug_image(image_path, gt_boxes, pred_boxes, pred_scores, pred_classes, output_filename):
    # 確保輸出檔名有 "pred_" 前綴
    out_dir = os.path.dirname(output_filename)
    out_name = os.path.basename(output_filename)
    if not out_name.startswith("pred_"):
        output_filename = os.path.join(out_dir, "pred_" + out_name)

    if not os.path.exists(image_path): return
    try:
        img_pil = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img_pil)
        w_img, h_img = img_pil.size
        
        # 這裡 scale 假設傳入的 boxes 是 0-256 的 Model Scale
        scale_x = w_img / 256.0
        scale_y = h_img / 256.0

        # 畫 GT (綠色) [cls, x1, y1, x2, y2]
        for box in gt_boxes:
            cls_id = int(box[0])
            x1 = box[1] * scale_x; y1 = box[2] * scale_y
            x2 = box[3] * scale_x; y2 = box[4] * scale_y
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)
            draw.text((x1, y1), f"GT:{cls_id}", fill="#00FF00")

        # 畫 Prediction (紅色) [x1, y1, x2, y2]
        for i, box in enumerate(pred_boxes):
            score = pred_scores[i]
            cls_id = int(pred_classes[i])
            if score < CONFIDENCE_THRESH: continue 
            x1 = box[0] * scale_x; y1 = box[1] * scale_y
            x2 = box[2] * scale_x; y2 = box[3] * scale_y
            draw.rectangle([x1, y1, x2, y2], outline="#FF0000", width=2)
            draw.text((x1, y1+30), f"P{cls_id}:{score:.2f}", fill="#FF0000")

        img_pil.save(output_filename)
        # print(f"    [視覺化] 圖片已儲存: {output_filename}")
    except Exception as e: print(f"存圖失敗: {e}")

def save_prediction_txt(output_path, pred_boxes, pred_classes, pred_scores):
    # 確保輸出檔名有 "pred_" 前綴
    out_dir = os.path.dirname(output_path)
    out_name = os.path.basename(output_path)
    if not out_name.startswith("pred_"):
        output_path = os.path.join(out_dir, "pred_" + out_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            for i, box in enumerate(pred_boxes):
                score = pred_scores[i]
                if score < CONFIDENCE_THRESH: continue
                
                cls_id = int(pred_classes[i])
                # 這裡的 box 是 0-256 scale (根據調用處決定)
                x1, y1, x2, y2 = box
                
                # 轉回 正規化 (0~1) 寫入 txt
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                
                ncx = cx / 256.0
                ncy = cy / 256.0
                nw = w / 256.0
                nh = h / 256.0
                
                f.write(f"{cls_id} {score:.6f} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")
    except Exception as e:
        print(f"寫入 TXT 失敗: {e}")

# ==============================================================================
# 4. 推論引擎 (YOLO LC)
# ==============================================================================
class YoloLCInference:
    def __init__(self, model_file):
        if not os.path.exists(model_file): raise FileNotFoundError(f"Model not found: {model_file}")
        
        self.interpreter = tflite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        
        self.input_index = self.input_details['index']
        self.output_index = self.output_details['index']
        
        self.nn_h = self.input_details['shape'][1]
        self.nn_w = self.input_details['shape'][2]
        self.input_dtype = self.input_details['dtype']
        
        # 量化參數
        self.in_scale, self.in_zp = self.input_details['quantization']
        self.out_scale, self.out_zp = self.output_details['quantization']

        print(f"Loaded Model: {os.path.basename(model_file)}")
        print(f"  Input: {self.nn_w}x{self.nn_h}, {self.input_dtype}")

        self.t_pre = 0.0; self.t_infer = 0.0; self.t_post = 0.0

    def preprocess(self, image_path):
        # 1. 讀取圖片 (PIL)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # 讀取並轉灰階
        img_pil = Image.open(image_path).convert('L') 
        t0 = time.perf_counter() 
        # 2. Resize (Stretch)
        img_resized = img_pil.resize((self.nn_w, self.nn_h))
        
        # 3. 轉為 Numpy array & 增加維度
        # PIL read as 0-255 uint8 by default
        input_data = np.asarray(img_resized, dtype=np.uint8)
        
        # Expand dims: (H, W) -> (H, W, 1) -> (1, H, W, 1)
        input_data = np.expand_dims(input_data, axis=-1)
        input_data = np.expand_dims(input_data, axis=0)

        self.t_pre = (time.perf_counter() - t0) * 1000
        return input_data

    def run_inference(self, input_data):
        t0 = time.perf_counter()
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_index)
        
        # 反量化 (若輸出不是 float)
        if self.out_scale > 0 and output_data.dtype != np.float32:
            output_data = (output_data.astype(np.float32) - self.out_zp) * self.out_scale
            
        self.t_infer = (time.perf_counter() - t0) * 1000
        return output_data

    def decode_and_nms(self, feats):
        t0 = time.perf_counter()
        
        # 1. 整理 Shape
        feats = np.squeeze(feats) # (16, 16, 50) or (16, 16, 10)
        
        # 自動適配 Channel 數
        total_channels = feats.shape[-1]
        features_per_anchor = 5 + NUM_CLASSES # 10
        num_anchors = total_channels // features_per_anchor
        
        # Reshape: (Grid, Grid, Anchors, 10)
        grid_h, grid_w = feats.shape[:2]
        feats = feats.reshape(grid_h, grid_w, num_anchors, features_per_anchor)
        
        # 2. 準備 Grid 與 Anchors
        grid_y, grid_x = np.mgrid[0:grid_h, 0:grid_w]
        grid_xy = np.stack((grid_x, grid_y), axis=-1).astype(np.float32)
        grid_xy = np.expand_dims(grid_xy, axis=2) # (16, 16, 1, 2)
        
        # 使用全域 ANCHORS，若數量不對則截斷或循環 (防呆)
        # 這裡假設您已經設置正確的 ANCHORS = [[0.14, 0.8]]
        anchors_tensor = np.array(ANCHORS[:num_anchors], dtype=np.float32).reshape(1, 1, num_anchors, 2)

        # 3. 核心解碼 (修正版)
        bx_by = sigmoid(feats[..., 0:2])
        # ★★★ 關鍵修正：寬高限制 exp 範圍，並除以 16.0 ★★★
        bw_bh = (np.exp(np.clip(feats[..., 2:4], -5.0, 5.0)) * anchors_tensor) / float(grid_w)
        
        # ★★★ 關鍵修正：中心點除以 16.0 ★★★
        pred_xy = (bx_by + grid_xy) / float(grid_w)
        
        # 4. 取得分數
        obj_conf = sigmoid(feats[..., 4:5])
        cls_scores = softmax(feats[..., 5:])
        final_scores = (obj_conf * cls_scores).reshape(-1, NUM_CLASSES)
        
        # bw_bh = bw_bh * 1.1 # 微調寬高以提升 IoU 表現
        # 5. 轉換為 (x1, y1, x2, y2) normalized
        # 展平
        pred_xy = pred_xy.reshape(-1, 2)
        pred_wh = bw_bh.reshape(-1, 2)
        
        # (cx, cy, w, h) -> (x1, y1, x2, y2)
        x1 = pred_xy[:, 0] - pred_wh[:, 0] / 2
        y1 = pred_xy[:, 1] - pred_wh[:, 1] / 2
        x2 = pred_xy[:, 0] + pred_wh[:, 0] / 2
        y2 = pred_xy[:, 1] + pred_wh[:, 1] / 2
        boxes = np.stack([x1, y1, x2, y2], axis=-1)
        
        # Clip to 0~1
        boxes = np.clip(boxes, 0.0, 1.0)

        # 6. NMS
        
        # 先做初步閾值過濾
        max_scores = np.max(final_scores, axis=1)
        keep_mask = max_scores >= CONFIDENCE_THRESH
        
        filtered_boxes = boxes[keep_mask]
        filtered_scores = max_scores[keep_mask]
        filtered_classes = np.argmax(final_scores[keep_mask], axis=1)
        
        if len(filtered_boxes) == 0:
            self.t_post = (time.perf_counter() - t0) * 1000
            return [], [], []

        # 轉成 OpenCV NMS 格式 [x, y, w, h] (0~1 scale for now is fine, or scale up)
        # NMSBoxes 建議使用 0-1 或 絕對坐標皆可，但要統一        
        indices = _non_max_suppression(filtered_boxes, filtered_scores, NMS_THRESH)
        
        final_boxes = []
        final_scores = []
        final_classes = []
        
        if len(indices) > 0:
            for i in indices: # 注意：這裡 indices 本身已經是可迭代的 numpy 陣列
                b = filtered_boxes[i]
                final_boxes.append([b[0]*256, b[1]*256, b[2]*256, b[3]*256]) 
                final_scores.append(filtered_scores[i])
                final_classes.append(filtered_classes[i])

        self.t_post = (time.perf_counter() - t0) * 1000
        return np.array(final_boxes), np.array(final_scores), np.array(final_classes)

# ==============================================================================
# 5. 評估與主流程
# ==============================================================================
def load_gt_labels(txt_path):
    """ 載入 GT 並轉為 [cls, x1, y1, x2, y2] (0-256 scale) """
    if not os.path.exists(txt_path): return []
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) >= 5:
                cls, cx, cy, w, h = parts[0:5]
                # 正規化 0~1 -> Model Scale 0~256
                cx *= 256; cy *= 256; w *= 256; h *= 256
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                boxes.append([int(cls), x1, y1, x2, y2])
    return boxes

def evaluate_single_image(pred_boxes, pred_classes, pred_scores, gt_boxes):
    if len(gt_boxes) == 0:
        # 如果模型有預測東西，這些全部都是 False Positive (is_tp=0)
        # 如果模型也沒預測東西，zip 會回傳空 list，這也是正確的
        stats = [(cls, score, 0) for cls, score in zip(pred_classes, pred_scores)]
        return stats, 0
    
    matched_gt = [False] * len(gt_boxes)
    # 依分數排序
    if len(pred_scores) > 0:
        sorted_idxs = np.argsort(pred_scores)[::-1]
    else:
        return [], len(gt_boxes)

    stats = [] # (class, score, is_tp)
    
    for idx in sorted_idxs:
        p_box = pred_boxes[idx] # [x1, y1, x2, y2]
        p_cls = pred_classes[idx]
        p_score = pred_scores[idx]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_boxes):
            gt_cls = gt[0]
            gt_box = gt[1:] # [x1, y1, x2, y2]
            
            if gt_cls == p_cls:
                iou = compute_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        if best_gt_idx >= 0 and best_iou >= EVAL_IOU_THRESH and not matched_gt[best_gt_idx]:
            matched_gt[best_gt_idx] = True
            stats.append((p_cls, p_score, 1)) # TP
        else:
            stats.append((p_cls, p_score, 0)) # FP
            
    return stats, len(gt_boxes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 預設路徑 (請修改為您實際的路徑)
    DEFAULT_MODEL = r"D:\Downloads\stm32MPU\Models\st_yolo_lc_v1\5class_best\fold1\0131_256\0131_YOLO_lc_fold1_PerTensor_quant_uint8_float32_random_1.tflite"
    DEFAULT_SOURCE = r"D:\Downloads\YOLO_Database\mit-bih_code\5_class_256\fold1\val"

    parser.add_argument("-m", "--model", required=False, default=DEFAULT_MODEL, help="Path to .tflite model")
    parser.add_argument("-s", "--source", required=False, default=DEFAULT_SOURCE, help="Path to image directory or file")
    args = parser.parse_args()
    
    OUTPUT_DIR = "lc_v1_tflite_results/1222_default_anchor"
    TXT_OUT_DIR = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TXT_OUT_DIR, exist_ok=True)

    # 1. 初始化模型
    engine = YoloLCInference(args.model)

    # 2. 準備圖片列表
    if os.path.isdir(args.source):
        img_files = glob.glob(os.path.join(args.source, "*.jpg"))
    else:
        img_files = [args.source]
    
    print(f"Starting inference on {len(img_files)} images...")

    # 統計變數
    tp = 0; fp = 0; total_gt = 0
    sum_pre = 0.0; sum_infer = 0.0; sum_post = 0.0
    all_pred_stats = [] # (class, score, is_tp)
    all_gt_counts = {}

    # 3. 主迴圈
    for i, img_path in enumerate(img_files):
        bn = os.path.splitext(os.path.basename(img_path))[0]
        
        # Preprocess
        input_data = engine.preprocess(img_path)
        
        # Infer
        raw_output = engine.run_inference(input_data)
        
        # Postprocess (Decode + NMS) -> returns 0-256 scale boxes
        p_boxes, p_scores, p_classes = engine.decode_and_nms(raw_output)
        
        # 累加時間
        sum_pre += engine.t_pre
        sum_infer += engine.t_infer
        sum_post += engine.t_post
        
       
        gt_txt_path = os.path.join(os.path.dirname(img_path), bn + ".txt")
        gt_boxes = load_gt_labels(gt_txt_path)
        
        # --- 存檔與視覺化 ---
        # 1. Save TXT
        # pred_txt_path = os.path.join(TXT_OUT_DIR, bn + ".txt")
        # if len(p_boxes) > 0:
        #     save_prediction_txt(pred_txt_path, p_boxes, p_classes, p_scores)
        # else:
        #     # 存空檔
        #     open(pred_txt_path, 'w').close()

        # # 2. Save Image (帶框)
        # pred_img_path = os.path.join(OUTPUT_DIR, f"pred_{bn}.jpg")
        # # 注意: p_boxes 已經是 0-256 scale，gt_boxes 也是，所以 save_debug_image 內部邏輯正確
        # save_debug_image(img_path, gt_boxes, p_boxes, p_scores, p_classes, pred_img_path)

        # --- 評估 (Evaluation) ---
        # 統計 GT 類別
        for g in gt_boxes:
            cid = int(g[0])
            all_gt_counts[cid] = all_gt_counts.get(cid, 0) + 1
            
        img_stats, n_gt = evaluate_single_image(p_boxes, p_classes, p_scores, gt_boxes)
        all_pred_stats.extend(img_stats)
        
        tp += sum([s[2] for s in img_stats])
        fp += len(img_stats) - sum([s[2] for s in img_stats])
        total_gt += n_gt

        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(img_files)}...")

    # ==============================================================================
    # 6. 計算與打印最終結果
    # ==============================================================================
    # 計算 Precision, Recall, F1 (Global)
    p = tp / (tp + fp + 1e-6)
    r = tp / (total_gt + 1e-6)
    f1 = 2 * p * r / (p + r + 1e-6)

    # 計算 mAP (Per Class AP then Mean)
    aps = []
    unique_classes = sorted(all_gt_counts.keys())
    
    for cls_id in unique_classes:
        n_gt_cls = all_gt_counts[cls_id]
        cls_preds = [x for x in all_pred_stats if x[0] == cls_id]
        
        if n_gt_cls == 0: continue
        if len(cls_preds) == 0:
            aps.append(0.0)
            continue
            
        cls_preds.sort(key=lambda x: x[1], reverse=True)
        tp_list = np.array([x[2] for x in cls_preds])
        fp_list = 1 - tp_list
        
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        
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
    if n > 0:
        print("\n=== Latency Analysis ===")
        print(f"Avg Pre-process:  {sum_pre/n:.4f} ms")
        print(f"Avg Inference:    {sum_infer/n:.4f} ms (CPU)") # TFLite 通常在 CPU 跑
        print(f"Avg Post-process: {sum_post/n:.4f} ms (Decode+NMS)")
        
        avg_total = (sum_pre + sum_infer + sum_post) / n
        print(f"Avg Total E2E:    {avg_total:.4f} ms")
        
        avg_fps = 1000.0 / avg_total if avg_total > 0 else 0
        print(f"System FPS:       {avg_fps:.2f} fps")
        print("========================================")