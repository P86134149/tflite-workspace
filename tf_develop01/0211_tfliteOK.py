import tensorflow as tf
import numpy as np
import pandas as pd

# 讀取 CSV 檔案
csv_path = r"C:\Users\wtmh\Downloads\senior_LEE\Model\tflite-workspace\tf_develop01\100m2_N.csv"
csv_pathN = r"C:\Users\wtmh\Downloads\senior_LEE\RAW DATA\1D\ECG\N\234m02631.csv"
csv_pathF = r"C:\Users\wtmh\Downloads\senior_LEE\RAW DATA\1D\ECG\F\233m02893.csv"
csv_pathQ = r"C:\Users\wtmh\Downloads\senior_LEE\RAW DATA\1D\ECG\Q\233m02893.csv"
csv_pathS = r"C:\Users\wtmh\Downloads\senior_LEE\RAW DATA\1D\ECG\S\233m02893.csv"
csv_pathV = r"C:\Users\wtmh\Downloads\senior_LEE\RAW DATA\1D\ECG\V\100m01907.csv"

data = pd.read_csv(csv_pathN, header=None)


# 載入 TFLite 模型
# model_path = r"C:\Users\wtmh\Downloads\senior_LEE\Model\1D\5\MIX_LMUEBCnet_1D_normalized_converted.tflite"
model_path = r"C:\Users\wtmh\Downloads\senior_LEE\Model\1D\5\MIX_LMUEBCnet_1D_normalized_PCQ_outputf32.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 獲取模型的輸入和輸出張量
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 根據模型的輸入型別調整資料型別
input_dtype = input_details[0]['dtype']
input_data = np.array(data.values, dtype=input_dtype)

# 準備輸入數據
input_data = np.resize(input_data, (256, 1))  # 調整數據形狀為 [256, 1]
input_data = np.expand_dims(input_data, axis=0)  # 添加批次維度，最終形狀為 [1, 256, 1]


# 設置模型輸入
interpreter.set_tensor(input_details[0]['index'], input_data)

# 執行推論
interpreter.invoke()

# 獲取模型輸出
output_data = interpreter.get_tensor(output_details[0]['index'])

# 定義標籤
ECG_labels = ["F", "Normal", "Q", "S", "V"]

# 獲取前3個分類結果
top_3_indices = np.argsort(output_data[0])[-3:][::-1]
top_3_scores = output_data[0][top_3_indices]

# 輸出結果
for i, (index, score) in enumerate(zip(top_3_indices, top_3_scores)):
    label = ECG_labels[index] if index < len(ECG_labels) else "Unknown"
    print(f"Top {i+1} class: {label}, score: {score}")