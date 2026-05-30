from __future__ import annotations

from collections import defaultdict

from vs_agent.models import Citation, MethodCard, RetrievalHit


def build_comparison_rows(hits: list[RetrievalHit]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for hit in hits:
        card = hit.card
        if not card or card.card_id in seen:
            continue
        seen.add(card.card_id)
        rows.append(
            {
                "论文": paper_display_name(hit.paper),
                "模型/模块": " / ".join(part for part in [card.model_name, card.module_type] if part) or card.title,
                "解决问题": card.problem_target or "-",
                "关联指标": ", ".join(card.metric_related) or "-",
                "可借鉴点": card.reusable_point,
                "来源": f"p.{card.page}" if card.page else "-",
            }
        )
    return rows


def build_brief(question: str, hits: list[RetrievalHit], citations: list[Citation]) -> str:
    cards = [hit.card for hit in hits if hit.card]
    by_paper: dict[str, list[MethodCard]] = defaultdict(list)
    for hit in hits:
        if hit.card:
            by_paper[paper_display_name(hit.paper)].append(hit.card)

    if not citations:
        return "当前文献库中未找到可靠来源，不能生成快速浏览综述。"

    lines = [
        f"基于当前文献库，和“{question}”相关的内容主要来自 {len(by_paper) or len(citations)} 篇论文。",
        "",
        "### 概览",
    ]
    for title, paper_cards in list(by_paper.items())[:6]:
        modules = sorted({card.module_type or card.model_name or card.title for card in paper_cards})
        lines.append(f"- {title}: {', '.join(modules)}。")

    if cards:
        lines.extend(["", "### 方法归纳"])
        for card in cards[:8]:
            cite = f"[p.{card.page}]" if card.page else "[source]"
            lines.append(f"- {card.title}: {card.evidence_summary} {cite}")

        lines.extend(["", "### 可借鉴点"])
        for card in cards[:6]:
            cite = f"[p.{card.page}]" if card.page else "[source]"
            lines.append(f"- {card.reusable_point} {cite}")

    return "\n".join(lines)


def paper_display_name(paper) -> str:
    return paper.original_filename or paper.filename_title or paper.title
