# 图像转 LaTeX：LoRA 微调与 RAG 检索增强

本项目的目标是把公式、表格等图片转换为 LaTeX 代码，并利用文档检索辅助生成注释。实验使用 Qwen2-VL-2B-Instruct 与 Qwen2.5-VL-7B-Instruct，比较原始模型、RAG、LoRA 以及 LoRA + RAG 四种方案。

## 项目组成

- `main_train/`：数据整理、训练配置、LoRA 微调与实验 Notebook。
- `RAG/`：PDF 检索、图像问答及 Streamlit 原型；具体入口见 [RAG/README.md](RAG/README.md)。
- `requirements.txt`：人工整理的直接依赖，不是原实验环境的完整锁定文件。
