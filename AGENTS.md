# AGENTS.md

本项目实现“虚拟染色个人科研模块知识库 Agent”。核心目标是用稳定、可演示、可维护的本地优先 MVP，完成论文入库、方法卡片、检索问答和来源引用闭环。

## 约束

- 页面第一屏就是工具，不做营销页。
- 每个回答必须能追溯到 PDF 原文。
- 方法卡片是核心资产，不能只做普通 chunk 检索。
- 没有来源时不编造回答。
- 默认可在无 API key 情况下运行；配置 OpenAI 兼容接口后可增强 LLM 抽取和回答。
- 本地数据保存在 `data/`，便于备份。

## 命令

```bash
pip install -r requirements.txt
streamlit run app.py
pytest
```

