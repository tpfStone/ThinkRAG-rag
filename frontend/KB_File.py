import os
import time

import pandas as pd
import streamlit as st

from server.utils.file import get_save_dir, save_uploaded_file


def list_existing_files():
    save_dir = get_save_dir()
    if not os.path.isdir(save_dir):
        return []

    files = []
    for name in sorted(os.listdir(save_dir)):
        path = os.path.join(save_dir, name)
        if os.path.isfile(path):
            files.append(
                {
                    "name": name,
                    "type": os.path.splitext(name)[1].lstrip(".") or "file",
                    "size": os.path.getsize(path),
                }
            )
    return files


def get_index_status():
    try:
        index_exists = st.session_state.index_manager.check_index_exists()
    except Exception as e:
        print(f"Failed to check index status: {type(e).__name__}: {e}")
        return {"exists": False, "document_count": 0, "node_count": 0}

    if not index_exists:
        return {"exists": False, "document_count": 0, "node_count": 0}

    docstore = st.session_state.index_manager.index.docstore
    ref_doc_info = docstore.get_all_ref_doc_info()
    unique_sources = set()
    for ref_doc_id, ref_doc in ref_doc_info.items():
        metadata = getattr(ref_doc, "metadata", {}) or {}
        source = metadata.get("file_path") or metadata.get("url_source") or metadata.get("title") or ref_doc_id
        unique_sources.add(source)

    return {
        "exists": True,
        "document_count": len(unique_sources),
        "node_count": len(docstore.docs),
    }


def render_file_table(files):
    st.dataframe(
        pd.DataFrame(files),
        column_config={
            "name": "File name",
            "size": st.column_config.NumberColumn("size", format="%d byte"),
            "type": "type",
        },
        hide_index=True,
    )


def render_quality_report(report, use_expander=True):
    if not report:
        return

    summary = report.get("summary", {})
    files = report.get("files", [])
    if not files:
        return

    bad_files = int(summary.get("bad_files", 0))
    warning_files = int(summary.get("warning_files", 0))
    indexable_documents = int(summary.get("indexable_documents", 0))
    skipped_documents = int(summary.get("skipped_documents", 0))
    parse_report = report.get("parse") or {}
    parse_candidate_pages = int(summary.get("parse_candidate_pages", parse_report.get("candidate_pages", 0) or 0))
    parse_repaired_pages = int(summary.get("parse_repaired_pages", parse_report.get("repaired_pages", 0) or 0))
    parse_failed_pages = int(summary.get("parse_failed_pages", parse_report.get("failed_pages", 0) or 0))

    if indexable_documents == 0:
        st.error("No indexable text was extracted. This file likely needs API parsing or a text-based PDF version.")
    elif bad_files > 0 or warning_files > 0:
        st.warning(
            "Document quality warnings found. "
            f"Indexable pages: {indexable_documents}; skipped pages: {skipped_documents}."
        )
    else:
        st.success(f"Document quality check passed. Indexable pages: {indexable_documents}.")

    if parse_candidate_pages > 0:
        if parse_repaired_pages > 0:
            st.info(
                "API document parsing repaired "
                f"{parse_repaired_pages}/{parse_candidate_pages} page(s); failed pages: {parse_failed_pages}."
            )
        elif parse_report.get("unavailable"):
            st.warning(
                "API document parsing was needed, but the parser is not available. "
                "Check `DASHSCOPE_API_KEY` and install `pip install -r requirements.txt`, then rebuild the index."
            )
        elif parse_failed_pages > 0:
            st.warning(f"API document parsing tried but did not repair any page. Failed pages: {parse_failed_pages}.")

    rows = []
    for item in files:
        rows.append(
            {
                "File": item.get("file_name", "unknown"),
                "Status": item.get("status", "unknown"),
                "Reason": item.get("reason", ""),
                "Pages": item.get("total_pages", 0),
                "Empty": item.get("empty_pages", 0),
                "Low quality": item.get("bad_pages", 0),
                "Avg text len": item.get("avg_text_len", 0),
                "Needs API parse": item.get("needs_api_parse", False),
                "Parse source": item.get("parse_source", "native"),
                "Parse status": item.get("parse_status", "success"),
                "Parse pages": item.get("parse_candidate_pages", 0),
                "Parse repaired": item.get("parse_repaired_pages", 0),
                "Parse failed": item.get("parse_failed_pages", 0),
                "Cache": item.get("parse_cache_hit", False),
            }
        )

    expanded = bad_files > 0 or warning_files > 0 or indexable_documents == 0
    if use_expander:
        with st.expander("Document quality report", expanded=expanded):
            _render_quality_report_details(rows, parse_report)
    else:
        st.caption("Document quality report")
        _render_quality_report_details(rows, parse_report)


def _render_quality_report_details(rows, parse_report):
    if parse_report.get("errors"):
        st.warning("Parse failure details: " + " | ".join(str(item) for item in parse_report["errors"][:3]))
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)


def _nodes_from_index_result(result):
    if isinstance(result, dict):
        return result.get("nodes", []) or []
    return result or []


def handle_index_result(result, render_report=True, report_use_expander=True):
    report = result.get("quality_report") if isinstance(result, dict) else None
    if report:
        st.session_state.last_quality_report = report
        if render_report:
            render_quality_report(report, use_expander=report_use_expander)

    nodes = _nodes_from_index_result(result)
    if not nodes:
        st.error("Knowledge base index was not generated because no valid text chunks were available.")
        return False

    summary = (report or {}).get("summary", {})
    indexed_docs = summary.get("indexable_documents", len(nodes))
    st.toast(f"Indexed {len(nodes)} chunks from {indexed_docs} indexable page(s).")
    return True


def build_index_from_data(
    chunk_size,
    chunk_overlap,
    spinner_text,
    render_report=True,
    rebuild=False,
    report_use_expander=True,
):
    print("Generating index from existing data directory...")
    with st.spinner(text=spinner_text):
        if rebuild:
            result = st.session_state.index_manager.rebuild_dir(get_save_dir(), chunk_size, chunk_overlap)
        else:
            result = st.session_state.index_manager.load_dir(get_save_dir(), chunk_size, chunk_overlap)
        if handle_index_result(result, render_report=render_report, report_use_expander=report_use_expander):
            st.toast("Knowledge base index generation complete")
            time.sleep(4)
            st.rerun()


def handle_file():
    st.header("Load Files")
    st.caption("Upload PDF, DOCX, TXT, and similar files, then index them into the knowledge base.")

    index_status = get_index_status()
    existing_files = list_existing_files()

    if index_status["exists"]:
        st.success(
            "Knowledge base index is ready. "
            f"Documents: {index_status['document_count']}; "
            f"Nodes: {index_status['node_count']}."
        )

    render_quality_report(st.session_state.get("last_quality_report"))

    with st.form("my-form", clear_on_submit=True):
        st.session_state.selected_files = st.file_uploader(
            "Upload files: ",
            accept_multiple_files=True,
            label_visibility="hidden",
        )
        submitted = st.form_submit_button(
            "Upload",
            help="Copy selected files into the local data directory.",
        )
        if len(st.session_state.selected_files) > 0 and submitted:
            print("Starting to upload files...")
            print(st.session_state.selected_files)
            for selected_file in st.session_state.selected_files:
                with st.spinner(f"Uploading {selected_file.name}..."):
                    save_dir = get_save_dir()
                    save_uploaded_file(selected_file, save_dir)
                    st.session_state.uploaded_files.append(
                        {"name": selected_file.name, "type": selected_file.type, "size": selected_file.size}
                    )
            st.toast("Upload successful")

    if len(st.session_state.uploaded_files) > 0:
        with st.expander("The following files are uploaded successfully.", expanded=True):
            render_file_table(st.session_state.uploaded_files)

    if len(st.session_state.uploaded_files) == 0 and existing_files:
        source_title = "Source files in data directory" if index_status["exists"] else "Files available to build the index"
        with st.expander(source_title, expanded=not index_status["exists"]):
            if index_status["exists"]:
                st.caption(
                    "These are source file copies. Query and BM25 retrieval use indexed text nodes in storage."
                )
            render_file_table(existing_files)

    with st.expander("Text Splitter Settings", expanded=True):
        cols = st.columns(2)
        chunk_size = cols[0].number_input(
            "Maximum length of a single text block: ",
            1,
            4096,
            st.session_state.chunk_size,
        )
        chunk_overlap = cols[1].number_input(
            "Adjacent text overlap length: ",
            0,
            st.session_state.chunk_size,
            st.session_state.chunk_overlap,
        )

    if len(st.session_state.uploaded_files) > 0:
        if st.button(
            "Index uploaded files",
            type="primary",
            help="Parse the uploaded files, generate embeddings, and save them to the knowledge base.",
        ):
            print("Generating index...")
            with st.spinner(text="Loading documents and building the index, may take a minute or two"):
                result = st.session_state.index_manager.load_files(
                    st.session_state.uploaded_files,
                    chunk_size,
                    chunk_overlap,
                )
                if handle_index_result(result):
                    st.toast("Knowledge base index generation complete")
                    st.session_state.uploaded_files = []
                    time.sleep(4)
                    st.rerun()

    if not index_status["exists"]:
        if st.button(
            "Build index from existing files",
            type="primary",
            disabled=len(st.session_state.uploaded_files) > 0 or len(existing_files) == 0,
            help="Use this after restarting the app when files already exist under the data directory.",
        ):
            build_index_from_data(
                chunk_size,
                chunk_overlap,
                "Loading existing documents and building the index, may take a minute or two",
            )
    elif existing_files:
        with st.expander("Advanced maintenance", expanded=False):
            st.warning(
                "Rebuilding reprocesses files in data and consumes embedding API quota. "
                "Use this only after changing chunk or embedding settings, or after repairing an index."
            )
            if st.button(
                "Rebuild index from data directory",
                disabled=len(st.session_state.uploaded_files) > 0,
                help="Advanced maintenance: re-ingest source files from the data directory.",
            ):
                build_index_from_data(
                    chunk_size,
                    chunk_overlap,
                    "Reprocessing existing documents, may take a minute or two",
                    render_report=True,
                    rebuild=True,
                    report_use_expander=False,
                )


handle_file()
