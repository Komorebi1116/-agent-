from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vs_agent.config import load_settings
from vs_agent.storage.sqlite_store import SQLiteStore
from vs_agent.ui.card_editor import render_card_editor
from vs_agent.ui.chat_panel import render_chat_panel
from vs_agent.ui.citation_panel import render_citation_panel
from vs_agent.ui.paper_panel import render_paper_panel


def main() -> None:
    st.set_page_config(page_title="虚拟染色科研 Agent", layout="wide")
    st.title("虚拟染色个人科研模块知识库 Agent")

    settings = load_settings()
    store = SQLiteStore(settings.db_path)

    if settings.llm_enabled:
        st.caption(
            f"LLM：{settings.openai_model}；Embedding：{settings.embedding_model}；接口：{settings.openai_base_url}"
        )
    else:
        st.caption("LLM：未配置密钥，当前使用本地规则抽卡与检索问答。")

    left, middle, right = st.columns([0.95, 1.55, 1.05], gap="medium")
    with left:
        render_paper_panel(settings, store)
    with middle:
        render_chat_panel(settings, store)
    with right:
        answer = st.session_state.get("last_answer")
        render_citation_panel(answer)
        render_card_editor(store, answer)


if __name__ == "__main__":
    main()
