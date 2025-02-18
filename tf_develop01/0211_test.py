import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import os

# 定義五類的資料夾路徑
base_path = r"C:\Users\wtmh\Downloads\senior_LEE\RAW DATA\1D\ECG"
categories = ["F", "N", "Q", "S", "V"]

# 讀取所有類別的 CSV 檔案
file_paths = {cat: glob.glob(os.path.join(base_path, cat, "*.csv")) for cat in categories}

# 載入 TFLite 模型
model_path = r"C:\Users\wtmh\Downloads\senior_LEE\Model\1D\5\MIX_LMUEBCnet_1D_normalized_PCQ_outputf32.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 獲取模型的輸入和輸出張量
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']
input_shape = input_details[0]['shape']

print(f"Model input shape: {input_shape}")

# 定義標籤
ECG_labels = ["F", "Normal", "Q", "S", "V"]

# 讀取並推論每個類別的數據
for cat, paths in file_paths.items():
    print(f"\nProcessing category: {cat} ({len(paths)} files found)")

    for csv_file in paths[:5]:  # 每類最多取 5 筆作為示範
        data = pd.read_csv(csv_file, header=None)
        input_data = np.array(data.values, dtype=input_dtype)

        # 調整形狀以符合模型要求
        input_data = np.resize(input_data, (256, 1))  # 轉換為 [256, 1]
        input_data = np.expand_dims(input_data, axis=0)  # 添加批次維度 [1, 256, 1]

        # 設置模型輸入
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # 執行推論
        interpreter.invoke()

        # 獲取模型輸出
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # 取得前3個分類結果
        top_3_indices = np.argsort(output_data[0])[-3:][::-1]
        top_3_scores = output_data[0][top_3_indices]

        # 輸出結果
        print(f"\nFile: {os.path.basename(csv_file)}")
        for i, (index, score) in enumerate(zip(top_3_indices, top_3_scores)):
            label = ECG_labels[index] if index < len(ECG_labels) else "Unknown"
            print(f"  Top {i+1} class: {label}, score: {score:.4f}")

