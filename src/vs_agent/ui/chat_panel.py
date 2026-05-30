from __future__ import annotations

import pandas as pd
import streamlit as st

from vs_agent.chat.answer_generator import AnswerGenerator
from vs_agent.config import Settings
from vs_agent.storage.sqlite_store import SQLiteStore


SAMPLE_QUESTIONS = [
    "哪些论文用了 CycleGAN 做虚拟染色？",
    "有哪些方法和 SSIM 提升有关？",
    "整理一下 attention 模块相关内容。",
]


def render_chat_panel(settings: Settings, store: SQLiteStore) -> None:
    st.subheader("文献问答")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    cols = st.columns(3)
    for col, question in zip(cols, SAMPLE_QUESTIONS):
        if col.button(question, use_container_width=True):
            _submit_question(settings, store, question)
            st.rerun()

    for message in st.session_state.messages[-8:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("table_rows"):
                st.dataframe(pd.DataFrame(message["table_rows"]), use_container_width=True, hide_index=True)

    question = st.chat_input("问你的个人文献库...")
    if question:
        _submit_question(settings, store, question)
        st.rerun()


def _submit_question(settings: Settings, store: SQLiteStore, question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    generator = AnswerGenerator(settings, store)
    with st.spinner("正在检索方法卡片和原文证据"):
        answer = generator.answer(question)
    st.session_state.last_answer = answer
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer.answer,
            "table_rows": answer.table_rows,
        }
    )

