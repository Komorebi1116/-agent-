from __future__ import annotations

from vs_agent.cards.paper_analyzer import PaperAnalysis, PaperAnalyzer
from vs_agent.config import Settings
from vs_agent.ingestion.paper_digest import build_paper_digest
from vs_agent.models import Chunk, MethodCard


class MethodCardExtractor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_warning: str | None = None
        self.last_error: str | None = None
        self.last_analysis: PaperAnalysis | None = None
        self.last_evidence_passed_count = 0
        self.last_generated_count = 0

    def extract(self, paper_id: str, chunks: list[Chunk], max_cards: int = 8) -> list[MethodCard]:
        self.last_warning = None
        self.last_error = None
        self.last_analysis = None
        self.last_evidence_passed_count = 0
        self.last_generated_count = 0

        if not self.settings.llm_enabled:
            self.last_error = "LLM未启用，禁止抽取方法卡片"
            raise RuntimeError(self.last_error)

        digest = build_paper_digest(paper_id, chunks)
        try:
            analyzer = PaperAnalyzer(self.settings)
            analysis = analyzer.analyze(digest)
            self.last_analysis = analysis
            cards = analyzer.extract_method_cards(digest, analysis, chunks, max_cards=max_cards)
        except Exception as exc:
            self.last_error = f"LLM方法卡片抽取失败，未生成规则卡片：{exc}"
            raise RuntimeError(self.last_error) from exc

        self.last_generated_count = len(cards)
        self.last_evidence_passed_count = len([card for card in cards if card.evidence_check_passed])
        if not cards:
            self.last_warning = "未抽取到高质量方法卡片，请检查 PDF 内容或 prompt"
        return cards

