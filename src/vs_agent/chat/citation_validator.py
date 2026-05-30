from __future__ import annotations

from vs_agent.models import ChatAnswer


def validate_answer(answer: ChatAnswer) -> ChatAnswer:
    if answer.answer.strip() and not answer.citations:
        answer.answer = (
            "当前文献库中未找到可靠来源，不能给出有依据的回答。"
            "请先上传相关论文，或换一个更具体的问题。"
        )
        answer.used_cards = []
        answer.used_chunks = []
        answer.related_cards = []
        answer.table_rows = []
    return answer

