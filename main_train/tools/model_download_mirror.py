import os
from modelscope import snapshot_download
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,

)
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, Qwen2VLForConditionalGeneration, AutoProcessor
import torch


import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"    # 镜像源
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"          # 启用高效传输协议
os.environ["HF_HOME"] = "./"  
            # 指定统一缓存目录


def model_download(model_name):

    if model_name == "Qwen/Qwen2-VL-2B-Instruct":
       model_dir = snapshot_download(model_name, cache_dir="./", revision="master")
     
    # 使用Transformers加载模型权重
       tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    # 特别的，Qwen2-VL-2B-Instruct模型需要使用Qwen2VLForConditionalGeneration来加载
       model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,)
       model.enable_input_require_grads() 

    elif model_name == "Qwen/Qwen2.5-VL-7B-Instruct":
         model_dir = snapshot_download(model_name, cache_dir="./", revision="master")
         tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
         model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
     model_name,
     torch_dtype=torch.bfloat16,
     attn_implementation="flash_attention_2",
     device_map="auto",
 )
         model.enable_input_require_grads() 
    
    return model, tokenizer, processor


def model_download_mirror(model_name):
    """支持多源下载的智能模型加载器"""
    logging.set_verbosity_info()
    
    # 模型源配置字典（可扩展）
    model_configs = {
        "Qwen/Qwen2-VL-2B-Instruct": {
            "source": "modelscope",
            "loader": {
                "processor": AutoProcessor,
                "model_class": Qwen2VLForConditionalGeneration,
                "model_args": {
                    "device_map": "auto",
                    "torch_dtype": torch.bfloat16,
                    "trust_remote_code": True
                },
                "tokenizer_args": {
                    "use_fast": False,
                    "trust_remote_code": True
                }
            }
        },
        
    }

    # 获取模型配置
    config = model_configs.get(model_name)
    if not config:
        raise ValueError(f"Unsupported model: {model_name}")

    try:
        print(f"🚀 正在下载模型: {model_name}...")
        
        # 选择下载源
        if config["source"] == "modelscope":
            local_path = snapshot_download(
                model_name, 
                cache_dir="./",
                revision="master"
            )
        else:  # huggingface
            local_path = hf_download(
                repo_id=model_name,
                cache_dir=os.getenv("HF_HOME", "./"),
                resume_download=True,
                endpoint="https://hf-mirror.com"  # 国内镜像
            )

        print(f"📦 模型已下载至: {local_path}")

        # 动态加载组件
        print("🔧 正在加载模型组件...")
        
        loader_config = config["loader"]
        
        # 加载processor
        processor = loader_config["processor"].from_pretrained(local_path)
        
        # 加载tokenizer（统一使用AutoTokenizer）
        tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            **loader_config["tokenizer_args"]
        )
        
        # 加载模型
        model = loader_config["model_class"].from_pretrained(
            local_path,
            **loader_config["model_args"]
        )

        print("✅ 模型加载完成")
        return model, tokenizer, processor

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise
 