from __future__ import annotations

import streamlit as st

from vs_agent.config import Settings
from vs_agent.ingestion.paper_ingestor import PaperIngestor
from vs_agent.storage.file_store import FileStore
from vs_agent.storage.sqlite_store import SQLiteStore


def render_paper_panel(settings: Settings, store: SQLiteStore) -> None:
    st.subheader("论文库")
    st.caption(
        f"LLM: {'enabled' if settings.llm_enabled else 'disabled'} | model: {settings.openai_model or '-'} | "
        f"base_url: {settings.openai_base_url} | API key: {'detected' if settings.openai_api_key else 'missing'} | "
        f"timeout: {settings.llm_timeout_seconds}s | retries: {settings.llm_max_retries}"
    )
    uploads = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    if uploads:
        file_store = FileStore(settings.papers_dir)
        ingestor = PaperIngestor(settings, store)
        for index, uploaded in enumerate(uploads):
            if st.button(f"入库 {uploaded.name}", key=f"ingest-{index}-{uploaded.name}"):
                status = st.status(f"正在入库 {uploaded.name}", expanded=True)
                try:
                    status.write("保存上传文件")
                    paper_id_hint = f"{index}_{abs(hash(uploaded.name))}"[:16]
                    path = file_store.save_upload(uploaded, uploaded.name, f"upload_{paper_id_hint}")
                    status.write("计算 MD5 并检查是否重复")
                    status.write("解析 PDF、切分 chunk、写入 SQLite 和 Chroma")
                    status.write("抽取方法卡片并写入 Chroma")
                    paper = ingestor.ingest_pdf(path)
                    if ingestor.last_duplicate:
                        status.update(label="该文献已入库，无需重复解析", state="complete")
                        st.info(f"该文献已入库，无需重复解析：{_paper_display_name(paper)}")
                    else:
                        status.update(label=f"已入库：{_paper_display_name(paper)}", state="complete")
                        if ingestor.last_card_extraction_warning:
                            st.warning(ingestor.last_card_extraction_warning)
                        st.caption(
                            f"卡片抽取方式：{ingestor.last_card_extraction_method} | "
                            f"LLM抽取耗时：{ingestor.last_timings.get('method_card_extract_seconds', 0):.2f}s | "
                            f"生成卡片数量：{ingestor.last_generated_card_count} | "
                            f"证据校验通过：{ingestor.last_evidence_passed_count}"
                        )
                        if ingestor.last_timings:
                            st.caption(_format_timings(ingestor.last_timings))
                    st.rerun()
                except Exception as exc:
                    status.update(label="入库失败", state="error")
                    error_text = ingestor.last_card_extraction_error or str(exc)
                    st.error(f"LLM方法卡片抽取失败，未生成规则卡片。原因：{error_text}")

    papers = store.list_papers()
    if not papers:
        st.info("还没有论文。先上传 1-3 篇 PDF 试运行。")
        return

    st.caption(f"{len(papers)} 篇论文")
    for paper in papers:
        with st.expander(_paper_display_name(paper), expanded=False):
            st.write(f"状态：`{paper.status}`")
            st.write(f"文献名称：{paper.filename_title or paper.title}")
            st.write(f"年份：{paper.year or '-'}")
            if paper.authors:
                st.write("作者：" + ", ".join(paper.authors[:6]))
            st.caption(paper.file_path)
            cards = store.list_cards(paper.paper_id)
            chunks = store.list_chunks(paper.paper_id)
            st.write(f"chunks：{len(chunks)}，方法卡片：{len(cards)}")


def _format_timings(timings: dict[str, float]) -> str:
    labels = {
        "md5_seconds": "MD5",
        "pdf_parse_seconds": "PDF解析",
        "chunk_split_seconds": "chunk切分",
        "chunk_sqlite_seconds": "chunk写SQLite",
        "chunk_chroma_seconds": "chunk写Chroma",
        "method_card_extract_seconds": "卡片抽取",
        "card_sqlite_seconds": "card写SQLite",
        "card_chroma_seconds": "card写Chroma",
    }
    return " | ".join(f"{labels.get(key, key)} {value:.2f}s" for key, value in timings.items())


def _paper_display_name(paper) -> str:
    return paper.original_filename or paper.filename_title or paper.title
