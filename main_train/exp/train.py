from datasets import Dataset, Features, Image, Value
from qwen_vl_utils import process_vision_info
from pathlib import Path
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, Qwen2VLForConditionalGeneration, AutoProcessor,AutoTokenizer

from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from functools import partial

import torch
import pandas as pd
import random
import swanlab
import json
import os

def model_train_func(model,output_dir,train_dataset,swanlab_callback,tokenizer):
    
    config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取LoRA模型
    train_peft_model = get_peft_model(model, config)
    
    # 配置训练参数
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        logging_steps=1,
        logging_first_step=1,
        num_train_epochs=2,
        save_steps=130,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
    model=train_peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

    
    return trainer


