from pathlib import Path
from types import SimpleNamespace

import pytest

import vs_agent.cards.card_extractor as card_extractor_module
from vs_agent.cards.card_extractor import MethodCardExtractor
from vs_agent.cards.paper_analyzer import PaperAnalysis, evidence_check
from vs_agent.models import Chunk, MethodCard


def settings(llm_enabled=True):
    return SimpleNamespace(
        llm_enabled=llm_enabled,
        project_root=Path(__file__).resolve().parents[1],
        openai_api_key="key" if llm_enabled else "",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_model="qwen-plus" if llm_enabled else "",
    )


def test_no_rule_based_fallback(monkeypatch):
    class FailingAnalyzer:
        def __init__(self, settings):
            pass

        def analyze(self, digest):
            raise RuntimeError("llm down")

    monkeypatch.setattr(card_extractor_module, "PaperAnalyzer", FailingAnalyzer)
    extractor = MethodCardExtractor(settings(llm_enabled=True))
    chunks = [Chunk("chunk_1", "paper_1", "Methods", 1, "Pix2pix uses a U-Net generator.")]

    with pytest.raises(RuntimeError, match="LLM方法卡片抽取失败"):
        extractor.extract("paper_1", chunks)

    assert extractor.last_error


def test_llm_disabled_raises():
    extractor = MethodCardExtractor(settings(llm_enabled=False))
    chunks = [Chunk("chunk_1", "paper_1", "Methods", 1, "Pix2pix uses a U-Net generator.")]

    with pytest.raises(RuntimeError, match="LLM未启用"):
        extractor.extract("paper_1", chunks)


def test_pix2pix_can_extract_cards(monkeypatch):
    class Pix2PixAnalyzer:
        def __init__(self, settings):
            pass

        def analyze(self, digest):
            return PaperAnalysis(paper_type="general image-to-image translation", main_task="paired translation")

        def extract_method_cards(self, digest, analysis, chunks, max_cards=8):
            return [
                _card("U-Net skip connections for spatial structure preservation", "generator_design", "chunk_1"),
                _card("PatchGAN discriminator for local texture realism", "discriminator_design", "chunk_1"),
                _card("L1 + adversarial loss for paired image translation", "loss_function", "chunk_1"),
            ]

    monkeypatch.setattr(card_extractor_module, "PaperAnalyzer", Pix2PixAnalyzer)
    extractor = MethodCardExtractor(settings(llm_enabled=True))
    chunks = [
        Chunk(
            "chunk_1",
            "paper_1",
            "Methods",
            2,
            "The pix2pix framework uses a U-Net generator with skip connections, a PatchGAN discriminator, and combines L1 loss with a conditional GAN objective.",
        )
    ]

    cards = extractor.extract("paper_1", chunks)

    assert [card.title for card in cards] == [
        "U-Net skip connections for spatial structure preservation",
        "PatchGAN discriminator for local texture realism",
        "L1 + adversarial loss for paired image translation",
    ]
    assert all(card.source == "llm" for card in cards)


def test_generic_title_rejected():
    chunk = Chunk("chunk_1", "paper_1", "Methods", 1, "The model uses a GAN objective for image translation.")
    card = _card("GAN", "architecture", "chunk_1")

    checked = evidence_check(card, {"chunk_1": chunk})

    assert not checked.evidence_check_passed
    assert "标题过泛" in checked.evidence_issue


def test_card_requires_evidence():
    chunk = Chunk("chunk_1", "paper_1", "Methods", 1, "The model uses a PatchGAN discriminator.")
    card = _card("PatchGAN discriminator for local texture realism", "discriminator_design", "chunk_1")
    card.source_chunk_ids = []
    card.evidence_quote = ""

    checked = evidence_check(card, {"chunk_1": chunk})

    assert not checked.evidence_check_passed
    assert "缺少 source_chunk_ids" in checked.evidence_issue
    assert "缺少 evidence_quote" in checked.evidence_issue


def test_llm_status_visible():
    s = settings(llm_enabled=True)
    assert s.llm_enabled
    assert s.openai_model == "qwen-plus"
    assert s.openai_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _card(title: str, card_type: str, chunk_id: str) -> MethodCard:
    return MethodCard(
        card_id=f"card_{title[:8].replace(' ', '_')}",
        paper_id="paper_1",
        title=title,
        card_type=card_type,
        task_type="paired image-to-image translation",
        core_problem="Need realistic image translation while preserving structure.",
        proposed_solution=title,
        technical_details="The method uses the cited pix2pix design components.",
        reusable_value_for_user_project="This design can transfer to virtual staining by preserving spatial tissue structure while improving realistic stain appearance.",
        evidence_quote="The pix2pix framework uses a U-Net generator with skip connections, a PatchGAN discriminator, and combines L1 loss with a conditional GAN objective.",
        evidence_page=2,
        source_chunk_ids=[chunk_id],
        confidence=0.9,
        evidence_check_passed=True,
        tags=["pix2pix"],
        source="llm",
        evidence_summary=title,
        reusable_point="This design can transfer to virtual staining by preserving spatial tissue structure.",
        source_quote="The pix2pix framework uses a U-Net generator with skip connections, a PatchGAN discriminator, and combines L1 loss with a conditional GAN objective.",
        page=2,
    )

