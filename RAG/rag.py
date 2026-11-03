# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,Qwen2_5_VLForConditionalGeneration,Qwen2_VLForConditionalGeneration,Qwen2_VLForConditionalGeneration,AutoProcessor
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

import torch, logging, os

# 开启调试模式和日志记录
set_debug(True)
set_verbose(True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)

class ChatPDF:
    """
    RAG over local PDF with a locally‑downloaded LLM & embedding model.
    本地化 RAG 系统，输入 PDF 后能回答相关问题
    """

    def __init__(
        self,
        llm_path: str = "/root/autodl-tmp/main_train/Qwen/Qwen2-VL-2B-Instruct",  # 本地 LLM 目录
        embed_path: str = "/root/autodl-tmp/RAG/mirror013/mxbai-embed-large-v1",   # 本地向量模型目录
        gpu: bool = True,
    ):
        # ------- 1) LLM --------
        logger.info(f"Loading LLM from {llm_path}")
        # 加载 tokenizer 和模型
        tok = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
        tok.pad_token = "<|eot_id|>"
        model = Qwen2_VLForConditionalGeneration.from_pretrained(
            llm_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(llm_path)
        
        gen_pipeline = pipeline(
            "image-to-text",
            model=model,
            tokenizer=tok,
            image_processor=processor,
            max_new_tokens=1024,
            framework="pt",
        )
        # 封装为 LangChain LLM
        self.model = HuggingFacePipeline(pipeline=gen_pipeline)

        # ------- 2) Embeddings --------
        # 向量模型加载部分
        logger.info(f"Loading embeddings from {embed_path}")
        # 加载本地嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_path,
            model_kwargs={"device": "cuda" if gpu else "cpu"},
        )

        # ------- 3) 其他组件 --------
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100  # 分块大小+重叠
        )
        # Prompt 模板设置
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a LaTeX OCR assistant. Your goal is to read the provided image and convert it into LaTeX code with line comments."),
            ("human", [
                {"type": "image_url", "image_url": "{image_path}"},
                {"text": "Context:\n{context}"},
                {"text": "Question: {question}"}
            ])
        ])
        

        # 初始化向量数据库和检索器
        self.vector_store = None
        self.retriever = None

    # ingest：加载并嵌入 PDF 内容
    def ingest(self, pdf_file_path: str):
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        # 加载 PDF 文档并按页读取内容
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        # 按块划分文档文本
        chunks = self.text_splitter.split_documents(docs)
        # 过滤掉复杂或无效的元信息（例如超大 metadict）
        chunks = filter_complex_metadata(chunks)

        # 构建 Chroma 向量数据库（并持久化）
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="chroma_db",
        )
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    # ask：执行检索并生成回答
    def ask(self, query: str, image_path: str, k: int = 5, score_threshold: float = 0.2):
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            # 初始化 Top-K 检索器（不设置阈值过滤）
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": k}
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)
        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        # 拼接上下文
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # 构建 Prompt
        formatted_input = {
        "image_path": image_path,
        "context": context,
        "question": query
    }

    # 构建正确的 RAG chain
        chain = (
            RunnablePassthrough.assign(context=lambda _: context)  # 传递上下文
            | self.prompt
            | self.model
            | StrOutputParser()
        )
        
        logger.info("Generating response using the LLM.")
        response = chain.invoke(formatted_input)
        # 截断 <END>
        if "<END>" in response:
            response = response.split("<END>")[0].strip()

        return response

    # clear：清除内存中的向量库
    def clear(self):
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None