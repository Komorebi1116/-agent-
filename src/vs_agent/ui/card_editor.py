from __future__ import annotations

import streamlit as st

from vs_agent.cards.card_service import CardService
from vs_agent.models import ChatAnswer, MethodCard
from vs_agent.storage.sqlite_store import SQLiteStore


def render_card_editor(store: SQLiteStore, answer: ChatAnswer | None) -> None:
    st.subheader("方法卡片")
    cards = answer.related_cards if answer and answer.related_cards else store.list_cards(include_ignored=False)[:10]
    if not cards:
        st.info("入库论文后会显示候选方法卡片。")
        return

    service = CardService(store)
    for card in cards:
        _render_card(card, service)


def _render_card(card: MethodCard, service: CardService) -> None:
    with st.expander(f"{card.title} · {card.user_status}", expanded=False):
        st.caption(
            f"source: {card.source} | type: {card.card_type or '-'} | evidence: {'passed' if card.evidence_check_passed else 'needs check'}"
        )
        st.write(card.proposed_solution or card.evidence_summary)
        if card.reusable_value_for_user_project:
            st.caption(card.reusable_value_for_user_project)
        if card.evidence_issue:
            st.warning(card.evidence_issue)
        meta = []
        if card.model_or_module or card.model_name:
            meta.append(card.model_or_module or card.model_name)
        if card.card_type or card.module_type:
            meta.append(card.card_type or card.module_type)
        if card.evaluation_metrics or card.metric_related:
            meta.extend(card.evaluation_metrics or card.metric_related)
        if meta:
            st.write(" / ".join(meta))
        st.caption(f"page: {card.evidence_page or card.page or '-'}")

        cols = st.columns(3)
        if cols[0].button("收藏", key=f"save-{card.card_id}"):
            service.set_status(card.card_id, "saved")
            st.rerun()
        if cols[1].button("待确认", key=f"candidate-{card.card_id}"):
            service.set_status(card.card_id, "candidate")
            st.rerun()
        if cols[2].button("忽略", key=f"ignore-{card.card_id}"):
            service.set_status(card.card_id, "ignored")
            st.rerun()

        with st.form(f"edit-{card.card_id}"):
            title = st.text_input("标题", value=card.title)
            tags = st.text_input("标签（逗号分隔）", value=", ".join(card.tags))
            submitted = st.form_submit_button("保存编辑")
            if submitted:
                service.rename(card.card_id, title)
                service.set_tags(card.card_id, tags)
                st.rerun()
