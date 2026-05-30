from __future__ import annotations

from vs_agent.cards.llm_client import LLMClient
from vs_agent.chat.brief_generator import build_brief, build_comparison_rows
from vs_agent.chat.citation_validator import validate_answer
from vs_agent.chat.intent_router import route_intent
from vs_agent.config import Settings
from vs_agent.models import ChatAnswer, Citation, MethodCard, RetrievalHit
from vs_agent.retrieval.retriever import Retriever
from vs_agent.storage.sqlite_store import SQLiteStore


class AnswerGenerator:
    def __init__(self, settings: Settings, store: SQLiteStore):
        self.settings = settings
        self.store = store
        self.retriever = Retriever(store, settings)
        self.llm = LLMClient(settings)
        self.qa_prompt = (settings.project_root / "src" / "vs_agent" / "prompts" / "research_qa.md").read_text(
            encoding="utf-8"
        )

    def answer(self, question: str) -> ChatAnswer:
        intent = route_intent(question)
        hits = self.retriever.retrieve(question, limit=10)
        citations = build_citations(hits)
        related_cards = [hit.card for hit in hits if hit.card][:8]
        used_cards = [card.card_id for card in related_cards]
        used_chunks = [hit.chunk.chunk_id for hit in hits if hit.chunk][:8]

        if not citations:
            return validate_answer(
                ChatAnswer(
                    answer="当前文献库中未找到可靠来源，不能给出有依据的回答。",
                    used_cards=[],
                    used_chunks=[],
                    citations=[],
                    related_cards=[],
                    intent=intent,
                )
            )

        if self.settings.llm_enabled:
            try:
                text = self._answer_with_llm(question, hits)
            except Exception:
                text = self._answer_with_rules(question, intent, hits, citations)
        else:
            text = self._answer_with_rules(question, intent, hits, citations)

        table_rows = build_comparison_rows(hits) if intent in {"metric_compare", "module_search", "model_brief"} else []
        return validate_answer(
            ChatAnswer(
                answer=text,
                used_cards=used_cards,
                used_chunks=used_chunks,
                citations=citations,
                related_cards=related_cards,
                intent=intent,
                table_rows=table_rows,
            )
        )

    def _answer_with_llm(self, question: str, hits: list[RetrievalHit]) -> str:
        evidence = format_evidence_pack(hits)
        return self.llm.chat_text(self.qa_prompt, f"Question:\n{question}\n\nEvidence pack:\n{evidence}")

    def _answer_with_rules(
        self,
        question: str,
        intent: str,
        hits: list[RetrievalHit],
        citations: list[Citation],
    ) -> str:
        if intent in {"model_brief", "metric_compare"}:
            return build_brief(question, hits, citations)

        card_lines = []
        chunk_lines = []
        for hit in hits:
            if hit.card:
                card = hit.card
                paper_name = paper_display_name(hit.paper)
                source = f"{paper_name}, p.{card.page}" if card.page else paper_name
                card_lines.append(f"- {card.title}: {card.evidence_summary} 来源：{source}")
            elif hit.chunk:
                paper_name = paper_display_name(hit.paper)
                source = f"{paper_name}, p.{hit.chunk.page}" if hit.chunk.page else paper_name
                chunk_lines.append(f"- {hit.chunk.text[:260].strip()} 来源：{source}")

        lines = ["基于当前文献库，找到以下有来源支持的内容："]
        if card_lines:
            lines.append("\n### 相关方法卡片")
            lines.extend(card_lines[:6])
        if chunk_lines:
            lines.append("\n### 原文证据")
            lines.extend(chunk_lines[:4])
        lines.append("\n这些结论仅来自已上传论文；如果需要更完整判断，建议继续上传相关论文。")
        return "\n".join(lines)


def build_citations(hits: list[RetrievalHit], limit: int = 8) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[tuple[str, int | None, str]] = set()
    for hit in hits:
        if hit.card:
            quote = hit.card.source_quote.strip()
            page = hit.card.page
            key = (hit.paper.paper_id, page, quote[:80])
            if quote and key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        paper_id=hit.paper.paper_id,
                        paper_title=paper_display_name(hit.paper),
                        page=page,
                        quote=quote,
                        card_id=hit.card.card_id,
                    )
                )
        elif hit.chunk:
            quote = hit.chunk.text[:800].strip()
            page = hit.chunk.page
            key = (hit.paper.paper_id, page, quote[:80])
            if quote and key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        paper_id=hit.paper.paper_id,
                        paper_title=paper_display_name(hit.paper),
                        page=page,
                        quote=quote,
                        chunk_id=hit.chunk.chunk_id,
                    )
                )
        if len(citations) >= limit:
            break
    return citations


def paper_display_name(paper) -> str:
    return paper.original_filename or paper.filename_title or paper.title


def format_evidence_pack(hits: list[RetrievalHit]) -> str:
    parts = []
    for index, hit in enumerate(hits, start=1):
        if hit.card:
            card = hit.card
            parts.append(
                f"[E{index}] type=method_card paper={paper_display_name(hit.paper)} page={card.page}\n"
                f"title={card.title}\nsummary={card.evidence_summary}\nquote={card.source_quote}"
            )
        elif hit.chunk:
            parts.append(
                f"[E{index}] type=chunk paper={paper_display_name(hit.paper)} page={hit.chunk.page}\n"
                f"text={hit.chunk.text[:1200]}"
            )
    return "\n\n".join(parts)
