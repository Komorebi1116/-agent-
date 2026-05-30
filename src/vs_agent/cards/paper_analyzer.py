from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field

from vs_agent.cards.card_utils import dedupe_method_cards
from vs_agent.cards.llm_client import LLMClient
from vs_agent.config import Settings
from vs_agent.ingestion.paper_digest import PaperDigest
from vs_agent.models import Chunk, MethodCard


CARD_TYPES = {
    "architecture",
    "generator_design",
    "discriminator_design",
    "loss_function",
    "training_strategy",
    "paired_translation",
    "unpaired_translation",
    "registration_alignment",
    "resolution_handling",
    "structure_preservation",
    "color_or_style_mapping",
    "evaluation_metric",
    "dataset_or_preprocessing",
    "general_image_translation",
    "virtual_staining_specific",
    "limitation_or_risk",
}
GENERIC_TITLES = {
    "gan",
    "cyclegan",
    "loss",
    "generator",
    "discriminator",
    "virtual staining",
    "attention",
    "transformer",
    "diffusion",
    "pix2pix",
}


@dataclass
class PaperAnalysis:
    paper_type: str = ""
    main_task: str = ""
    input_modality: str = ""
    output_modality: str = ""
    data_setting: str = ""
    core_methods: list[str] = field(default_factory=list)
    losses: list[str] = field(default_factory=list)
    evaluation_metrics: list[str] = field(default_factory=list)
    relevance_to_user_project: str = ""
    relevance_reason: str = ""
    useful_aspects: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict:
        return self.__dict__


class PaperAnalyzer:
    def __init__(self, settings: Settings):
        if not settings.llm_enabled:
            raise RuntimeError("LLM未启用，禁止抽取方法卡片")
        self.settings = settings
        self.llm = LLMClient(settings)
        prompt_dir = settings.project_root / "src" / "vs_agent" / "prompts"
        self.analysis_prompt = (prompt_dir / "paper_analysis.md").read_text(encoding="utf-8")
        self.card_prompt = (prompt_dir / "method_card_extraction_v2.md").read_text(encoding="utf-8")
        self.evidence_prompt = (prompt_dir / "method_card_evidence_check.md").read_text(encoding="utf-8")

    def analyze(self, digest: PaperDigest) -> PaperAnalysis:
        prompt_payload = compact_json(digest.to_prompt_dict())
        try:
            data = self.llm.chat_json(self.analysis_prompt, prompt_payload)
        except Exception as exc:
            raise RuntimeError(f"论文理解阶段 LLM 调用失败，prompt_chars={len(prompt_payload)}：{exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("论文理解阶段返回的 JSON 不是对象")
        return PaperAnalysis(
            paper_type=str(data.get("paper_type") or ""),
            main_task=str(data.get("main_task") or ""),
            input_modality=str(data.get("input_modality") or ""),
            output_modality=str(data.get("output_modality") or ""),
            data_setting=str(data.get("data_setting") or ""),
            core_methods=_as_str_list(data.get("core_methods")),
            losses=_as_str_list(data.get("losses")),
            evaluation_metrics=_as_str_list(data.get("evaluation_metrics")),
            relevance_to_user_project=str(data.get("relevance_to_user_project") or ""),
            relevance_reason=str(data.get("relevance_reason") or ""),
            useful_aspects=_as_str_list(data.get("useful_aspects")),
        )

    def extract_method_cards(
        self,
        digest: PaperDigest,
        analysis: PaperAnalysis,
        chunks: list[Chunk],
        max_cards: int = 8,
    ) -> list[MethodCard]:
        payload = {
            "paper_digest": digest.to_prompt_dict(),
            "paper_analysis": analysis.to_prompt_dict(),
            "max_cards": max_cards,
        }
        prompt_payload = compact_json(payload)
        try:
            data = self.llm.chat_json(self.card_prompt, prompt_payload)
        except Exception as exc:
            raise RuntimeError(f"方法卡片抽取阶段 LLM 调用失败，prompt_chars={len(prompt_payload)}：{exc}") from exc
        raw_cards = data.get("cards", []) if isinstance(data, dict) else []
        if not isinstance(raw_cards, list):
            raise RuntimeError("方法卡片抽取阶段返回的 cards 不是列表")

        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        cards: list[MethodCard] = []
        issues: list[str] = []
        for raw in raw_cards:
            if not isinstance(raw, dict):
                continue
            card = self._raw_to_card(digest.paper_id, raw)
            checked = evidence_check(card, chunk_by_id)
            if not checked.evidence_check_passed:
                issues.append(f"{checked.title}: {checked.evidence_issue}")
                continue
            cards.append(checked)

        return dedupe_method_cards(cards, max_cards=max_cards)

    def _raw_to_card(self, paper_id: str, raw: dict) -> MethodCard:
        card_type = str(raw.get("card_type") or "").strip()
        if card_type not in CARD_TYPES:
            card_type = "general_image_translation"
        source_chunk_ids = _as_str_list(raw.get("source_chunk_ids"))
        evidence_page = _as_int(raw.get("evidence_page"))
        return MethodCard(
            card_id=str(raw.get("card_id") or f"card_{uuid.uuid4().hex[:16]}"),
            paper_id=paper_id,
            title=str(raw.get("title") or "").strip(),
            card_type=card_type,
            task_type=str(raw.get("task_type") or "").strip(),
            input_output=_none_if_blank(raw.get("input_output")),
            core_problem=str(raw.get("core_problem") or "").strip(),
            proposed_solution=str(raw.get("proposed_solution") or "").strip(),
            model_or_module=_none_if_blank(raw.get("model_or_module")),
            technical_details=str(raw.get("technical_details") or "").strip(),
            loss_design=_none_if_blank(raw.get("loss_design")),
            training_setting=_none_if_blank(raw.get("training_setting")),
            evaluation_metrics=_as_str_list(raw.get("evaluation_metrics")),
            why_it_works=_none_if_blank(raw.get("why_it_works")),
            reusable_value_for_user_project=str(raw.get("reusable_value_for_user_project") or "").strip(),
            limitation=_none_if_blank(raw.get("limitation")),
            evidence_quote=str(raw.get("evidence_quote") or "").strip(),
            evidence_page=evidence_page,
            source_chunk_ids=source_chunk_ids,
            confidence=_as_float(raw.get("confidence"), default=0.0),
            evidence_check_passed=False,
            evidence_issue=None,
            tags=_as_str_list(raw.get("tags")),
            user_status=str(raw.get("user_status") or "candidate"),
            source="llm",
            task=str(raw.get("task_type") or ""),
            model_name=_none_if_blank(raw.get("model_or_module")),
            module_type=card_type,
            problem_target=str(raw.get("core_problem") or ""),
            metric_related=_as_str_list(raw.get("evaluation_metrics")),
            evidence_summary=str(raw.get("proposed_solution") or ""),
            reusable_point=str(raw.get("reusable_value_for_user_project") or ""),
            source_quote=str(raw.get("evidence_quote") or "").strip(),
            page=evidence_page,
        )


def evidence_check(card: MethodCard, chunk_by_id: dict[str, Chunk]) -> MethodCard:
    issues: list[str] = []
    if not is_specific_title(card.title):
        issues.append("标题过泛")
    if not card.source_chunk_ids:
        issues.append("缺少 source_chunk_ids")
    if not card.evidence_quote:
        issues.append("缺少 evidence_quote")
    if not card.proposed_solution:
        issues.append("缺少 proposed_solution")
    if len(card.reusable_value_for_user_project.strip()) < 20:
        issues.append("reusable_value_for_user_project 不够具体")

    source_text = " ".join(chunk_by_id[cid].text for cid in card.source_chunk_ids if cid in chunk_by_id)
    if card.source_chunk_ids and not source_text:
        issues.append("source_chunk_ids 不存在")
    if card.evidence_quote and source_text and not quote_supported(card.evidence_quote, source_text):
        issues.append("evidence_quote 未在对应 chunk 中找到")
    if card.evidence_quote and card.proposed_solution and not has_overlap(card.evidence_quote, card.proposed_solution):
        issues.append("evidence_quote 对 proposed_solution 支持不足")

    card.evidence_check_passed = not issues
    card.evidence_issue = "; ".join(issues) if issues else None
    if card.evidence_page is None:
        for cid in card.source_chunk_ids:
            chunk = chunk_by_id.get(cid)
            if chunk and chunk.page is not None:
                card.evidence_page = chunk.page
                card.page = chunk.page
                break
    card.source = "llm"
    return card


def is_specific_title(title: str) -> bool:
    clean = re.sub(r"\s+", " ", title.strip()).lower()
    if clean in GENERIC_TITLES:
        return False
    return len(clean.split()) >= 3 or ("+" in clean and len(clean.split()) >= 2)


def quote_supported(quote: str, source_text: str) -> bool:
    quote_norm = normalize_text(quote)
    source_norm = normalize_text(source_text)
    if quote_norm in source_norm:
        return True
    quote_tokens = meaningful_tokens(quote_norm)
    if not quote_tokens:
        return False
    source_tokens = set(meaningful_tokens(source_norm))
    return len([token for token in quote_tokens if token in source_tokens]) / len(quote_tokens) >= 0.6


def has_overlap(left: str, right: str) -> bool:
    left_tokens = set(meaningful_tokens(left))
    right_tokens = set(meaningful_tokens(right))
    if not left_tokens or not right_tokens:
        return False
    return len(left_tokens & right_tokens) >= 2


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def meaningful_tokens(text: str) -> list[str]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "using", "use", "our", "are"}
    return [token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+-]+", text.lower()) if token not in stop and len(token) > 2]


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _as_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _none_if_blank(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
