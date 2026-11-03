# 图像转 LaTeX：LoRA 微调与 RAG 课程项目

本项目是自然语言处理课程小组作业，目标是把公式、表格等图片转换为 LaTeX 代码，并利用文档检索辅助生成注释。实验使用 Qwen2-VL-2B-Instruct 与 Qwen2.5-VL-7B-Instruct，比较原始模型、RAG、LoRA 以及 LoRA + RAG 四种方案。

## 项目组成

- `main_train/`：数据整理、训练配置、LoRA 微调与实验 Notebook。
- `RAG/`：PDF 检索、图像问答及 Streamlit 原型；具体入口见 [RAG/README.md](RAG/README.md)。
- `requirements.txt`：人工整理的直接依赖，不是原实验环境的完整锁定文件。

## 数据、模型与运行

课程实验使用公开数据集 ArXiv-tables、LaTeX_OCR-full 与 datikz。数据图片、生成的 CSV/JSON、基座模型、LoRA checkpoint、向量库和下载缓存均留在本机，并由 `.gitignore` 排除。克隆仓库后需要自行获取数据及模型，并遵守其各自许可。

原实验在 Linux 云 GPU 环境完成。Notebook 从 `main_train/` 目录运行；配置文件与部分 RAG 代码仍包含实验时的 `/root/autodl-tmp/` 路径。在新机器上运行前，需要按本机路径调整配置和数据标注文件。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

训练流程可从 `main_train/main_train.ipynb`（2B）或 `main_train/main_train-7b.ipynb`（7B）阅读。`main_train/main_train.py` 是早期探索脚本，不作为主要训练入口。运行训练需要 GPU；仓库尚未验证最低显存需求或在全新环境中完成端到端复现。

## 实验结论与限制

本地课程报告 `report.pdf` 记录了数据处理、实验设置及不同模型与 LoRA/RAG 组合的示例对比。复杂公式、表格仍可能出现结构和符号错误。仓库未提供独立测试集上的标准化公开基准，因此不把示例结果描述为通用准确率。报告包含小组成员信息和第三方资料，默认不提交到公开仓库。

## 小组分工与致谢

项目由三名组员共同完成。按课程报告记录，祝天扬参与数据集搜集、云服务器与环境配置、部分微调代码搭建与测试，以及报告撰写；模型训练、RAG 与 Streamlit 等其他工作由小组协作完成。RAG 原型参考了 [chatpdf-rag-deepseek-r1](https://github.com/paquino11/chatpdf-rag-deepseek-r1) 的 PDF 检索交互思路。
