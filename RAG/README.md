# RAG 与 Streamlit 原型

本目录包含课程项目中用于 PDF 检索、图片输入和 LaTeX 生成的实验原型。代码是在 Linux 云 GPU 环境中开发的研究脚本，仍含实验时的绝对路径；运行前需要准备 Qwen-VL 基座模型、对应 LoRA checkpoint、向量模型和可用于检索的 PDF。

## 文件

- `app.py` / `rag.py`：早期 Streamlit + RAG 原型。
- `app1.py` / `rag1.py`：多模态输入原型。
- `app1_trained.py` / `rag1_trained.py`：加载 LoRA adapter 的实验版本。

在准备模型与路径之后，可从本目录尝试 `streamlit run app1_trained.py`。原代码的路径和依赖版本尚未在全新环境中完成端到端验证；若仅需理解实现，先阅读 `rag1_trained.py` 中模型、检索与推理流程。向量数据库、下载的嵌入模型及本地参考 PDF 已在根目录 `.gitignore` 中排除。

本模块参考 [chatpdf-rag-deepseek-r1](https://github.com/paquino11/chatpdf-rag-deepseek-r1) 的 PDF 检索交互思路，原仓库的说明不直接适用于这里的 Qwen-VL + LoRA 实验。
