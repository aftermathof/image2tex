# rag.py
import base64
import logging
import os
import torch
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores.utils import filter_complex_metadata

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    pipeline
)

from qwen_vl_utils import process_vision_info
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatPDF:
    """支持多模态输入的本地RAG系统"""
    
    def __init__(
        self,
        llm_path: str = "/root/autodl-tmp/main_train/Qwen/Qwen2.5-VL-7B-Instruct",
        embed_path: str = "/root/autodl-tmp/RAG/mirror013/mxbai-embed-large-v1",
        output_dir = "/root/autodl-tmp/main_train/output/Qwen_formula_LatexOCR",

        gpu: bool = True,
        max_new_tokens: int = 50000000000
    ):
        """
        初始化RAG系统
        
        :param llm_path: 本地LLM模型路径
        :param embed_path: 嵌入模型路径
        :param gpu: 是否使用GPU
        :param max_new_tokens: 最大生成token数
        """
        self.llm_path = llm_path
        self.embed_path = embed_path
        self.output_dir=output_dir
        self.load_model_path=f"{output_dir}/checkpoint-{max([int(d.split('-')[-1]) for d in os.listdir(output_dir) if d.startswith('checkpoint-')])}",
        self.gpu = gpu
        self.max_new_tokens = max_new_tokens
        
        # 初始化组件
        self._init_llm()
        self._init_embeddings()
        self._init_prompt_template()
        self._init_text_processor()
        
        # 向量数据库
        self.vector_store = None
        self.retriever = None

    def _init_llm(self):
        """初始化多模态LLM"""
        logger.info("Initializing Qwen-VL model...")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_path, 
            use_fast=True,
            pad_token="<|endoftext|>"
        )
        
        # 加载模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.llm_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
     
        )
       
        val_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=True,  # 训练模式
            r=32,  # Lora 秩
            lora_alpha=64,  # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.05,  # Dropout 比例
            bias="none",
        )
        
        print(f"load_model_path: {self.load_model_path}")
        self.model=PeftModel.from_pretrained(model, model_id=self.load_model_path[0], config=val_config)
        # 图像处理器
        self.processor = AutoProcessor.from_pretrained(self.llm_path)
        
        # 构建pipeline
        self.gen_pipeline = pipeline(
            "image-to-text",
            model=self.model,
            tokenizer=self.tokenizer,
            image_processor=self.processor,
            max_new_tokens=self.max_new_tokens,

        )
        
        # 封装为LangChain组件
        self.langchain_llm = HuggingFacePipeline(pipeline=self.gen_pipeline)

    def _init_embeddings(self):
        """初始化文本嵌入模型"""
        logger.info("Initializing embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_path,
            model_kwargs={"device": "cuda" if self.gpu else "cpu"},
            encode_kwargs={'normalize_embeddings': True}
        )

    def cleanup_temp_files(self):
        """清理临时文件"""
        if hasattr(self, '_temp_files'):
            for path in self._temp_files:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        logger.info(f"已清理临时文件: {path}")
                except Exception as e:
                    logger.warning(f"清理文件失败 {path}: {str(e)}")
            self._temp_files = []

    def _init_prompt_template(self):
        """构建多模态提示模板"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",  "你是一个专业的 LaTeX 助理。目标是读取用户输入的照片，转换成LaTex的表格公式，并提供详细的注释。"),
            HumanMessage(
                content=[
                    {"type": "image", "image": "{image_base64}"}
                    ,
                    {"type": "text", "text": (
                        "文档上下文：{context}"
                        "用户问题：{question}"
                        "回答要求："
                        "- 生成有效的 LaTeX 代码"
                        "- 为每个部分添加行注释进行解释"
                    )}
                ]
            )
        ])

    def _init_text_processor(self):
        """初始化文本处理器"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )
    def image_to_base64(self,image_path:str)->str:
        return self._image_to_base64(image_path)
    def _image_to_base64(self, image_path: str) -> str:
        """转换本地图像为Base64数据URL"""
        try:
            with open(image_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode("utf-8")
                return encoded
        except Exception as e:
            logger.error(f"图像处理失败: {str(e)}")
            raise RuntimeError(f"无法读取图像文件: {image_path}") from e

    def ingest(self, pdf_path: str):
        """
        处理并存储PDF文档
        
        :param pdf_path: PDF文件路径
        """
        logger.info(f"开始处理文档: {pdf_path}")
        torch.cuda.empty_cache()
        
        try:
            # 加载文档
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # 分块处理
            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)
            
            # 创建向量库
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                # persist_directory="chroma_db"
            )
            
            # 初始化检索器
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k":5}
            )
            
            logger.info("文档处理完成")
            
        except Exception as e:
            logger.error(f"文档处理失败: {str(e)}")
            raise RuntimeError("文档处理失败，请检查文件格式") from e

    def ask(self, query: str, image_base64: str) -> str:
        """
        执行多模态问答
        
        :param query: 用户问题
        :param image_base64: base64编码的图片数据（需包含data:前缀）
        :return: 生成的回答
        """
        logger.info(f"处理查询: {query}")
        
        # 验证状态
        if not self.retriever:
            raise ValueError("请先加载文档")
        
        try:
            # 步骤1：检索上下文
            context_docs = self.retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in context_docs)
            # print(image_base64)
            # 步骤2：直接使用传入的base64数据
            # inputs = {
            #     "image_base64": image_base64,  # 直接使用参数值
            #     "context": context,
            #     "question": query
            # }
            
            # # 步骤3：构建处理链
            # chain = (
            #     RunnablePassthrough()
            #     | self.prompt
            #     | self.langchain_llm
            #     | StrOutputParser()
            # )

            messages = [
            {"role": "system", "content": "你是一个专业的 LaTeX 助理。目标是读取用户输入的照片，转换成LaTex的表格公式，并提供详细的注释。"},
            {"role": "user",
            "content":
                [
                    {"type": "image", "image_url": f"data:image/jpeg;base64,{image_base64}"},
                    # {"type": "image", "image": "{image_base64}"},
                    {"type": "text", "text": (
                        f"文档上下文：{context}"
                        f"用户问题：{query}"
                        "回答要求："
                        "- 生成有效的 LaTeX 代码"
                        "- 为每个部分添加行注释进行解释"
                    )}
                ]
            }
        ]
            
            # # 步骤4：生成响应
            # response = chain.invoke(inputs)
            
            # # 后处理
            # return self._postprocess(response)
            text = self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda")
            generated_ids = self.model.generate(**inputs, max_new_tokens=50000)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print("output:",output_text)

            return output_text[0]





        except torch.cuda.OutOfMemoryError:
            self._clean_gpu_memory()
            return "内存不足，请尝试简化问题或缩小图像尺寸"
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"生成失败: {str(e)}")
            return f"生成错误: {str(e)}"



    def _clean_gpu_memory(self):
        """清理GPU内存"""
        logger.warning("执行GPU内存清理")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
    def clear(self):
        """重置系统状态"""
        logger.info("清理系统资源")
        self._clean_gpu_memory()
        self.vector_store = None
        self.retriever = None


#unit test

if __name__=="__main__":
    torch.cuda.empty_cache()
    assistant = ChatPDF()
    filename= "/root/autodl-tmp/RAG/LaTeX使用手册.pdf"
    assistant.ingest(filename)
    prompt = "Hi, please help me."

    image_path = "/root/autodl-tmp/RAG/0001.png"

    image_base64= assistant.image_to_base64(image_path)

    response = assistant.ask(prompt,image_base64=image_base64)
   
    # from PIL import Image
    # from io import BytesIO
    # image = "data:image/jpeg;base64,"+image_base64
    # _, base64_data = image.split("base64,", 1)
    # data = base64.b64decode(base64_data)
    # image_obj = Image.open(BytesIO(data))




