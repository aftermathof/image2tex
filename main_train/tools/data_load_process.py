#from modelscope.msdatasets import MsDataset
import os
import pandas as pd
from datasets import load_dataset
import random
import json
from datasets import Dataset, Features, Image, Value
from qwen_vl_utils import process_vision_info
from pathlib import Path
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, Qwen2VLForConditionalGeneration, AutoProcessor,AutoTokenizer
import torch


from tools import process, model_download_localload, model_download_mirror, data_load_process
import os
import random
import json


def data_csv(dataset_dir,csv_path,ds):
    total = len(ds)

        # 创建保存图片的目录
    os.makedirs(dataset_dir, exist_ok=True)

        # 初始化存储图片路径和描述的列表
    image_paths = []
    texts = []

    # 在循环开始前定义最大长度限制（可根据需要调整）
    max_code_length = 4096  # 例如设置最大允许512个字符

    for i in range(total):
        # 获取每个样本的信息
        item = ds[i]
        text = item['code']
        
        # ====== 新增：文本长度筛选 ======
        if len(text) > max_code_length:
            print(f"Skipping sample {i} (code length {len(text)} > {max_code_length})")
            continue
        # ====== 筛选结束 ======
        
        image = item['image']
        
        # 保存图片并记录路径
        image_path = os.path.abspath(f'{dataset_dir}/{i}.jpg')
        image.save(image_path)
        
        # 将路径和描述添加到列表中
        image_paths.append(image_path)
        texts.append(text)
        
        # 每处理50张图片打印一次进度（注意现在显示的是实际处理的进度）
        if (len(image_paths) % 50 == 0) and (len(image_paths) > 0):
            processed_count = len(image_paths)
            print(f'Processing {processed_count}/{total} valid images ({(processed_count)/total*100:.1f}%)')

            # 将图片路径和描述保存为CSV文件
    df = pd.DataFrame({
                'image_path': image_paths,
                'text': texts,
            })

            # 将数据保存为CSV文件
    df.to_csv(csv_path, index=False)
    print(f'数据处理完成，共处理了{total}张图片')
   
    return



def load_picture():

    dataset_id = 'AI-ModelScope/LaTeX_OCR'
    subset_name = 'default'
    dataset_dir = 'formula_LaTeX_OCR'
    csv_path = './formula_latex_ocr_train.csv'


# 检查目录是否已存在
    if not os.path.exists(dataset_dir):
        # 从modelscope下载COCO 2014图像描述数据集
        ds =  MsDataset.load(dataset_id , subset_name=subset_name)
        ds1=ds["train"]
        ds2=ds["test"]
        ds3=ds["validation"]
        ds = concatenate_datasets([
        ds1,
        ds2,
        ds3
    ])
        
    data_csv(dataset_dir,csv_path,ds)
    
    return


def load_formula():

    dataset_id = "HuggingFaceM4/datikz"
    dataset_dir = 'Picture_LaTeX_OCR'
    csv_path = './Picture_latex_ocr_train.csv'
    
                     
    
        # 从modelscope下载COCO 2014图像描述数据集
    ds = load_dataset(dataset_id)
    ds=ds["train"]
        # 设置处理的图片数量上限
    data_csv(dataset_dir,csv_path,ds)

    return


def conversation_generator(csv_path,train_json_path,test_json_path,val_json_path):
# Create conversation format
        df = pd.read_csv(csv_path,encoding='ibm866')
        
        conversations = []
        
        # Add image conversations
        for i in range(len(df)):
            conversations.append({
                "id": f"identity_{i+1}",
                "conversations": [
                    {
                        "role": "user",
                        "value": f"{df.iloc[i]['image_path']}"
                    },
                    {
                        "role": "assistant", 
                        "value": str(df.iloc[i]['text'])
                    }
                ]
            })
        random.shuffle(conversations)

# 计算划分点
        total = len(conversations)
        train_split = int(total * 0.8)
        test_split = int(total * 0.9)
        
        # 划分数据集
        train_data = conversations[:train_split]
        test_data = conversations[train_split:test_split]
        val_data = conversations[test_split:]
    
       
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
            
        with open(test_json_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        # Save validation set 
        
        with open(val_json_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        return

