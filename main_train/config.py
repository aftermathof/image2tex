# config.py
# 模型配置
model_name = "Qwen/Qwen2-VL-2B-Instruct"
model_base_dir = "Qwen/Qwen2-VL-2B-Instruct"
output_dir = "./output/Qwen_Picture_LatexOCR"

# 数据配置
train_dataset_json_path = "Picture_latex_ocr_train.json"
csv_path = './Picture_latex_ocr_train.csv'
train_json_path = './Picture_latex_ocr_train.json'
test_json_path = './Picture_latex_ocr_test.json'
val_json_path = './Picture_latex_ocr_val.json'

# 提示语配置
prompt = "你是一个LaTeX OCR助手,目标是读取用户输入的照片，转换成LaTeX公式。"