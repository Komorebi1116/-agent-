from vs_agent.models import Chunk, MethodCard, Paper
from vs_agent.retrieval.embedder import LocalHashEmbedder, create_embedder
from vs_agent.retrieval.retriever import Retriever
from vs_agent.retrieval.chroma_store import ChromaVectorStore
from vs_agent.storage.sqlite_store import SQLiteStore


class LocalSettings:
    remote_embedding_enabled = False
    embedding_model = "local-hash-512"
    chroma_dir = None

    def __init__(self, chroma_dir):
        self.chroma_dir = chroma_dir


def test_retriever_returns_card_and_chunk_hits(tmp_path):
    store = SQLiteStore(tmp_path / "app.db")
    paper = Paper("paper_1", "CycleGAN Virtual Staining", [], "2024", "paper.pdf", "now", "processed")
    chunk = Chunk("chunk_1", "paper_1", "Methods", 2, "CycleGAN improves virtual staining and SSIM.")
    card = MethodCard(
        card_id="card_1",
        paper_id="paper_1",
        title="CycleGAN structural loss for virtual staining consistency",
        card_type="loss_function",
        task_type="unpaired translation",
        core_problem="structure consistency",
        proposed_solution="CycleGAN with structural loss is evaluated by SSIM.",
        technical_details="Structural loss constrains virtual staining output.",
        reusable_value_for_user_project="Use structural loss for SSIM-related structure preservation in virtual staining.",
        evidence_quote="CycleGAN improves virtual staining and SSIM.",
        evidence_page=2,
        source_chunk_ids=["chunk_1"],
        evaluation_metrics=["SSIM"],
        evidence_check_passed=True,
        source="llm",
        tags=["CycleGAN", "SSIM"],
        user_status="saved",
        task="virtual staining",
        model_name="CycleGAN",
        module_type="loss_function",
        problem_target="structure consistency",
        metric_related=["SSIM"],
        evidence_summary="CycleGAN with structural loss is evaluated by SSIM.",
        reusable_point="Use structural loss for SSIM-related structure preservation.",
        source_quote="CycleGAN improves virtual staining and SSIM.",
        page=2,
    )
    store.upsert_paper(paper)
    store.upsert_chunks([chunk])
    store.upsert_cards([card])
    settings = LocalSettings(tmp_path / "chroma")
    vector_store = ChromaVectorStore(settings, LocalHashEmbedder())
    vector_store.add_chunks([chunk])
    vector_store.add_cards([card])

    hits = Retriever(store, settings).retrieve("SSIM CycleGAN", limit=5)

    assert hits
    assert any(hit.card and hit.card.card_id == "card_1" for hit in hits)


def test_local_hash_embedder_supports_batch():
    embedder = LocalHashEmbedder()

    embeddings = embedder.embed_batch(["CycleGAN", "SSIM"])

    assert len(embeddings) == 2
    assert len(embeddings[0]) == embedder.dimensions


def test_create_embedder_uses_remote_when_configured():
    class Settings:
        remote_embedding_enabled = True
        embedding_model = "text-embedding-v4"
        openai_api_key = "sk-test"
        openai_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    embedder = create_embedder(Settings())

    assert embedder.name == "text-embedding-v4"
