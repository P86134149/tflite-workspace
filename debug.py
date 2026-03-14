import cv2
import numpy as np
import tensorflow as tf
import os

# ================= 設定區 =================
# 請替換成您的一張測試圖片路徑
TEST_IMG_PATH = r"D:\Downloads\YOLO_Database\mit-bih_code\5_class\test\234_352.jpg"
MODEL_PATH = r"D:\Downloads\stm32MPU\Models\st_yolo_lc_v1\5FOLD_best\fold1\1222_default_anchor\1222YOLOlc_fold1_PerTensor_quant_uint8_float32_random_1.tflite"

# 您的 Anchors
ANCHORS = [
  [0.076023, 0.258508],
  [0.163031, 0.413531],
  [0.234769, 0.702585],
  [0.427054, 0.715892],
  [0.748154, 0.857092]
]
# =========================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def draw_debug_boxes():
    # 1. 載入模型
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. 圖片預處理
    img_bgr = cv2.imread(TEST_IMG_PATH)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_orig, w_orig = img_gray.shape
    
    # Resize & Expand dims
    img_resized = cv2.resize(img_gray, (256, 256))
    input_data = np.expand_dims(img_resized, axis=-1)
    input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)

    # 3. 推論
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index']) # (1, 16, 16, 50)

    # 4. 手動解碼 (Verbose Debugging)
    print(f"Analyzing output for {TEST_IMG_PATH}...")
    
    # Reshape
    # 假設排列: [Batch, GridY, GridX, Anchor, (tx, ty, tw, th, conf, cls...)]
    reshaped = np.reshape(output_data, (1, 16, 16, 5, 10))
    
    # 取得 Grid
    grid_x, grid_y = np.meshgrid(np.arange(16), np.arange(16))
    grid_x = np.reshape(grid_x, (1, 16, 16, 1))
    grid_y = np.reshape(grid_y, (1, 16, 16, 1))

    # 取出各個分量
    tx = reshaped[..., 0]
    ty = reshaped[..., 1]
    tw = reshaped[..., 2]
    th = reshaped[..., 3]
    conf_logits = reshaped[..., 4]
    class_logits = reshaped[..., 5:] # shape (..., 5)

    # 計算數值
    scores = sigmoid(conf_logits) * np.max(sigmoid(class_logits), axis=-1)
    
    # 找出分數最高的前 5 個位置
    flat_scores = scores.flatten()
    top_indices = np.argsort(flat_scores)[-5:][::-1] # 取最大的5個
    
    print(f"Top 5 raw scores: {flat_scores[top_indices]}")

    vis_img = cv2.resize(img_bgr, (512, 512)) # 放大一點方便看
    
    for i, idx in enumerate(top_indices):
        # 反解回 (gy, gx, anchor_idx)
        # idx 是 flat index，維度是 (16, 16, 5) -> 16*16*5 = 1280
        # numpy reshape order is typically C-style (last dim changes fastest)
        # idx = gy * (16*5) + gx * 5 + a
        
        a_idx = idx % 5
        rem = idx // 5
        gx = rem % 16
        gy = rem // 16
        
        # 取得該點的 raw values
        this_tx = tx[0, gy, gx, a_idx]
        this_ty = ty[0, gy, gx, a_idx]
        this_tw = tw[0, gy, gx, a_idx]
        this_th = th[0, gy, gx, a_idx]
        
        # 計算 Box (Standard YOLO)
        bx = (sigmoid(this_tx) + gx) / 16.0
        by = (sigmoid(this_ty) + gy) / 16.0
        bw = np.exp(this_tw) * ANCHORS[a_idx][0] /16.0
        bh = np.exp(this_th) * ANCHORS[a_idx][1] /16.0
        
        print(f"Top {i+1}: Grid({gx},{gy}) Anchor[{a_idx}]")
        print(f"  Raw: tx={this_tx:.2f}, ty={this_ty:.2f}, tw={this_tw:.2f}, th={this_th:.2f}")
        print(f"  Decoded(Norm): x={bx:.2f}, y={by:.2f}, w={bw:.2f}, h={bh:.2f}")

        # 畫在圖上
        x1 = int((bx - bw/2) * 512)
        y1 = int((by - bh/2) * 512)
        x2 = int((bx + bw/2) * 512)
        y2 = int((by + bh/2) * 512)
        
        color = (0, 255, 0) # Green
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_img, f"#{i+1}", (x1, y1+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    output_file = "debug_vis.jpg"
    cv2.imwrite(output_file, vis_img)
    print(f"\nDebug image saved to: {output_file}")
    print("請查看這張圖片。")
    print("1. 如果框的位置是對的但大小非常大 -> 可能是 Anchor 順序錯了或是 exp() 爆炸。")
    print("2. 如果框都在左上角 -> 可能是 Grid offset 沒加對。")
    print("3. 如果框的長寬比明顯相反 (例如應該寬卻變高) -> 可能是 w, h 順序相反。")

if __name__ == "__main__":
    draw_debug_boxes()