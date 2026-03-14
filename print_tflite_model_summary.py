import argparse
import os

# 嘗試引入 TFLite Interpreter
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Error: 請安裝 tflite-runtime 或 tensorflow (pip install tensorflow)")
        exit(1)

def print_model_summary(model_path):
    """
    載入 TFLite 模型並打印簡化的模型架構摘要。
    注意: TFLite 模型是優化的圖形結構，無法像 Keras 一樣提供完整的層級摘要。
    此函數使用 get_tensor_details() 來列出所有張量資訊，並模擬類似 Keras 的格式。
    額外打印輸入和輸出細節，以提供更多資訊。
    """
    if not os.path.exists(model_path):
        print(f"Error: 模型檔案不存在: {model_path}")
        return
    
    print(f"載入 TFLite 模型: {model_path}")
    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # 獲取輸入和輸出細節
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    print("=" * 50)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"  Model: {model_name}")
    print("=" * 50)
    print("  Layer index  Trainable    Name                    Type                  Params#  Output shape")
    
    # 首先打印輸入張量
    for detail in input_details:
        index = detail['index']
        name = detail['name'][:30]
        shape = str(detail['shape'])
        dtype = str(detail['dtype'])
        trainable = "False"
        params = "N/A"
        layer_type = "Input Tensor"
        print(f"  {index:<12} {trainable:<10} {name:<22} {layer_type:<20} {params:<8} {shape}")
    
    # 打印輸出張量
    for detail in output_details:
        index = detail['index']
        name = detail['name'][:30]
        shape = str(detail['shape'])
        dtype = str(detail['dtype'])
        trainable = "False"
        params = "N/A"
        layer_type = "Output Tensor"
        print(f"  {index:<12} {trainable:<10} {name:<22} {layer_type:<20} {params:<8} {shape}")
    
    # 打印其他張量（如果有）
    for i, detail in enumerate(tensor_details):
        index = detail['index']
        # 跳過已經打印的輸入和輸出
        if any(index == inp['index'] for inp in input_details) or any(index == out['index'] for out in output_details):
            continue
        name = detail['name'][:30]
        shape = str(detail['shape'])
        dtype = str(detail['dtype'])
        trainable = "False"
        params = "N/A"
        layer_type = "Tensor"
        print(f"  {index:<12} {trainable:<10} {name:<22} {layer_type:<20} {params:<8} {shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="打印 TFLite 模型架構摘要")
    parser.add_argument("-m", "--model", required=True, help="TFLite 模型檔案路徑")
    args = parser.parse_args()
    
    print_model_summary(args.model)