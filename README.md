# RAG-ms

一个面向中文企业报告场景的 RAG 问答项目，支持 PDF 报告解析、文本分块、向量检索、重排和大模型生成答案。

## 项目特点

- 支持 PDF 报告解析与结构化处理
- 支持中文企业年报/研报场景
- 支持检索增强生成（RAG）问答流程
- 支持向量检索与重排
- 支持本地脚本和 Streamlit 界面运行
- 兼容国内模型与接口调用

## 目录说明

- `main.py`：主流程入口
- `app_streamlit.py`：Streamlit 界面
- `src/`：核心功能代码
- `data/stock_data/`：示例数据与问答结果
- `requirements.txt`：依赖列表

## 环境准备

建议使用 Python 3.10+。

安装依赖：

```bash
pip install -r requirements.txt
```

## 运行方式

### 1. 运行主流程

```bash
python main.py
```

### 2. 启动界面

```bash
streamlit run app_streamlit.py
```

## 数据说明

示例数据位于 `data/stock_data/`，包含：

- `questions.json` / `questions-1.json`
- `answers_*.json`
- `subset.csv`
- `pdf_reports.xlsx`

## 备注

- 如果你在 Git 合并时遇到 `README.md` 冲突，可以直接用这份简洁版覆盖。
- 项目中的具体模型配置、API Key 和数据路径，请根据自己的环境调整。
