# 医学图像科研 AI Assistant

面向医学图像算法研究的文献解析、知识沉淀与智能问答助手。

这个项目是一个可本地运行的 Streamlit AI 产品 Demo，支持 PDF 论文导入、ChromaDB 向量检索、LLM 论文理解抽取方法卡、带来源引用的问答，以及医学图像算法论文推荐。

仓库内已经包含一组演示数据：2 篇示例论文 PDF、SQLite 业务库和 ChromaDB 向量索引。克隆后直接启动即可看到论文、方法卡和引用面板，不会是空白知识库。

## 功能概览<img width="2546" height="1402" alt="d42bbb924fb425abb54394df30b8e678" src="https://github.com/user-attachments/assets/0a0b9b11-0ba7-4866-9611-215fec843ec1" />


- PDF 论文上传、解析、MD5 去重和入库。
- 使用 ChromaDB 做 chunk 与方法卡向量检索。
- 使用 OpenAI-compatible API 调用大模型抽取方法卡，不再使用规则 fallback。
- 方法卡包含核心问题、解决思路、关键模块、损失设计、可迁移价值、证据引用和页码。
- 问答只基于已导入论文回答，并显示可追溯证据。
- “问一问 / 高频使用方法 / 最近新增论文”三类工作台入口。
- 最近新增论文按医学图像算法与深度学习相关性筛选。

## 快速启动

建议使用 Python 3.10 或 3.11。

```bash
pip install -r requirements.txt
streamlit run app.py
```

启动后浏览器打开：

```text
http://localhost:8501
```

Windows 也可以使用脚本启动：

```powershell
.\scripts\run_streamlit.ps1
```

## API 配置

项目默认读取系统或用户环境变量 `OPENAI_API_KEY`。不要把真实密钥提交到仓库。

默认模型配置如下：

```text
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-plus
OPENAI_EMBEDDING_MODEL=text-embedding-v4
OPENAI_TIMEOUT_SECONDS=180
OPENAI_MAX_RETRIES=2
```

如果需要本地覆盖配置，可以复制示例文件：

```bash
copy .env.example .env
```

然后在 `.env` 中填写：

```text
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-plus
OPENAI_EMBEDDING_MODEL=text-embedding-v4
```

注意：方法卡抽取是 LLM-only。未配置 API key、LLM 调用失败或证据不足时，系统会明确提示失败，不会静默生成规则卡片。

## 演示数据

仓库自带的演示数据位于：

```text
data/app.db
data/chroma/
data/papers/
```

其中包含 pix2pix 与 CycleGAN 两篇示例论文，方便打开后直接演示知识库、方法卡和引用来源。

如果想清空本地演示数据并重新导入自己的 PDF，可以运行：

```bash
python scripts/reset_local_data.py
```

## 项目结构

```text
app.py
requirements.txt
data/
  app.db
  chroma/
  papers/
scripts/
src/vs_agent/
  cards/
  chat/
  feed/
  ingestion/
  prompts/
  recommend/
  retrieval/
  storage/
  ui/
tests/
```

## 常用命令

运行测试：

```bash
pytest
```

导出演示指标：

```bash
python scripts/export_demo_metrics.py
```

清空本地数据：

```bash
python scripts/reset_local_data.py
```

## 隐私说明

本项目默认面向本地 Demo 和个人科研使用。上传的 PDF、解析后的 chunk、方法卡和向量索引会保存在本地 `data/` 目录。若论文包含敏感信息，请谨慎启用远程 LLM，并避免把包含敏感内容的 `data/` 目录提交到公开仓库。
