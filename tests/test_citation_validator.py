from vs_agent.chat.citation_validator import validate_answer
from vs_agent.models import ChatAnswer


def test_validator_rejects_answer_without_citations():
    answer = ChatAnswer(
        answer="CycleGAN works well.",
        used_cards=["card_1"],
        used_chunks=[],
        citations=[],
        related_cards=[],
    )

    validated = validate_answer(answer)

    assert "未找到可靠来源" in validated.answer
    assert validated.used_cards == []

