from modelscope import snapshot_download
from transformers import (TrainingArguments, 
Trainer,
AutoTokenizer,
DataCollatorForSeq2Seq,
Qwen2VLForConditionalGeneration, 
AutoProcessor,
AutoModelForVisualQuestionAnswering)
import requests



from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch

def model_download(model_name):
    if model_name=="Qwen/Qwen2-VL-2B-Instruct":
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./", use_fast=False, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,)
    
       
    return model,tokenizer,processor



from pathlib import Path
from transformers import AutoTokenizer
import os

def load_model_from_local(model_base_dir):
    try:

        if model_base_dir == "Qwen/Qwen2-VL-2B-Instruct":
           tokenizer = AutoTokenizer.from_pretrained(model_base_dir, use_fast=False, trust_remote_code=True)
           processor = AutoProcessor.from_pretrained(model_base_dir)
           model = Qwen2VLForConditionalGeneration.from_pretrained(model_base_dir, 
                                                                    device_map="auto",
                                                                    torch_dtype=torch.bfloat16,            
                                                                    trust_remote_code=True,)
           model.enable_input_require_grads()
         
        elif model_base_dir == "Qwen/Qwen2.5-VL-7B-Instruct":
           tokenizer = AutoTokenizer.from_pretrained(model_base_dir, use_fast=False, trust_remote_code=True)
           processor = AutoProcessor.from_pretrained(model_base_dir)
            
            
           model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_base_dir, 
                                                                    device_map="auto",
                                                                    torch_dtype=torch.bfloat16,            
                                                                    trust_remote_code=True,)
           model.enable_input_require_grads()
        print("🎉 模型加载成功")
        return model, tokenizer, processor
        
    except Exception as e:
        print(f"加载失败，请检查：\n"
              f"1. 文件权限: chmod -R 755 {base_model_dir}\n"
              f"2. PyTorch版本: pip install torch>=2.0\n"
              f"3. 完整错误信息:\n{str(e)}")
        raise

# 使用示例 ----------------------------------------------------


