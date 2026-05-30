from __future__ import annotations

import streamlit as st

from vs_agent.models import ChatAnswer


def render_citation_panel(answer: ChatAnswer | None) -> None:
    st.subheader("来源引用")
    if not answer or not answer.citations:
        st.info("提问后这里会显示论文、页码和原文片段。")
        return

    for index, citation in enumerate(answer.citations, start=1):
        with st.expander(f"{index}. {citation.paper_title}", expanded=index == 1):
            st.write(f"页码：{citation.page or '-'}")
            st.write(citation.quote)

