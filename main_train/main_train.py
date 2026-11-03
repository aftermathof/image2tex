from datasets import Dataset, Features, Image, Value
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from modelscope import snapshot_download
from tools import process, model_download,model_download_mirror



model_name="Salesforce/blip2-flan-t5-xl"
train_dataset_json_path = "Picture_latex_ocr_train.json"
model_folder = "models--Salesforce--blip2-flan-t5-xl"


if os.path.exists(model_folder):
    print("模型文件夹已存在，从本地加载模型...")
    model = AutoModel.from_pretrained(model_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    processor = AutoFeatureExtractor.from_pretrained(model_folder)
else:
    print("模型文件夹不存在，正在下载模型...")
    model, tokenizer, processor = model_download_mirror.model_download_mirror(model_name)


print(model)

train_ds = Dataset.from_json(train_dataset_json_path)
subset = train_ds.select(range(1000)).map(process.process_func)