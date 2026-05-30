from vs_agent.ingestion.chunker import chunk_pages
from vs_agent.ingestion.pdf_parser import ParsedPage


def test_chunk_pages_keeps_page_mapping():
    pages = [
        ParsedPage(
            page=3,
            text=(
                "Methods\n\nWe use CycleGAN for virtual staining. "
                "The generator includes attention modules and structural loss. "
            )
            * 12,
        )
    ]

    chunks = chunk_pages("paper_1", pages)

    assert chunks
    assert all(chunk.paper_id == "paper_1" for chunk in chunks)
    assert all(chunk.page == 3 for chunk in chunks)
    assert "CycleGAN" in chunks[0].text

