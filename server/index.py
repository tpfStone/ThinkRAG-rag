# Index management - create, load and insert
import os
import shutil
from datetime import datetime
from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core import load_index_from_storage, load_indices_from_storage

from config import DEV_MODE, STORAGE_DIR
from server.document_parsing import list_input_files, parse_input_files
from server.document_quality import build_quality_report, prepare_documents_for_indexing
from server.ingestion import AdvancedIngestionPipeline
from server.stores.strage_context import STORAGE_CONTEXT
from server.utils.file import get_save_dir
from server.utils_json import sanitize_for_json


INDEX_STORAGE_FILES = {
    "default__vector_store.json",
    "docstore.json",
    "graph_store.json",
    "image__vector_store.json",
    "index_store.json",
    "property_graph_store.json",
    "vector_store.json",
}
STAGING_STORAGE_DIR = f"{STORAGE_DIR}_staging"


class IndexManager:
    def __init__(self, index_name):
        self.index_name: str = index_name
        self.storage_context: StorageContext = STORAGE_CONTEXT
        self.persist_dir: str = STORAGE_DIR
        self.index_id: str = None
        self.index: VectorStoreIndex = None

    def check_index_exists(self):
        indices = load_indices_from_storage(self.storage_context)
        print(f"Loaded {len(indices)} indices")
        if len(indices) > 0:
            self.index = indices[0]
            self.index_id = indices[0].index_id
            return True
        return False

    def init_index(self, nodes):
        self.index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            store_nodes_override=True,
        )
        self.index_id = self.index.index_id
        if DEV_MODE:
            self.storage_context.persist(persist_dir=self.persist_dir)
        print(f"Created index {self.index.index_id}")
        return self.index

    def load_index(self):
        if self.index is not None:
            print(f"Index {self.index.index_id} already loaded")
            return self.index

        if self.index_id is not None:
            self.index = load_index_from_storage(self.storage_context, index_id=self.index_id)
        else:
            try:
                self.index = load_index_from_storage(self.storage_context)
            except ValueError as e:
                indices = load_indices_from_storage(self.storage_context)
                if len(indices) > 0:
                    self.index = indices[0]
                    self.index_id = indices[0].index_id
                else:
                    raise ValueError("No indices found in storage context. Please create an index first.") from e

        if not DEV_MODE:
            self.index._store_nodes_override = True
        print(f"Loaded index {self.index.index_id}")
        return self.index

    def insert_nodes(self, nodes):
        if self.index is not None:
            self.index.insert_nodes(nodes=nodes)
            if DEV_MODE:
                self.storage_context.persist(persist_dir=self.persist_dir)
            print(f"Inserted {len(nodes)} nodes into index {self.index.index_id}")
        else:
            self.init_index(nodes=nodes)
        return self.index

    def load_dir(self, input_dir, chunk_size, chunk_overlap, use_ocr=None):
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        files = list_input_files(input_dir)
        if not files:
            print("No documents found")
            return {"nodes": [], "quality_report": build_quality_report([])}
        parsed = parse_input_files(files, chunk_size, chunk_overlap)
        return self._ingest_parsed_files(parsed)

    def rebuild_dir(self, input_dir, chunk_size, chunk_overlap, use_ocr=None):
        if not DEV_MODE:
            raise RuntimeError("Full rebuild staging is only implemented for development file storage.")
        return self._rebuild_dir_with_staging(input_dir, chunk_size, chunk_overlap)

    def load_files(self, uploaded_files, chunk_size, chunk_overlap, use_ocr=None):
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        save_dir = get_save_dir()
        files = [os.path.join(save_dir, file["name"]) for file in uploaded_files]
        print(files)
        parsed = parse_input_files(files, chunk_size, chunk_overlap)
        return self._ingest_parsed_files(parsed)

    def load_websites(self, websites, chunk_size, chunk_overlap):
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        from server.readers.beautiful_soup_web import BeautifulSoupWebReader

        if isinstance(websites, str):
            websites = [u.strip() for u in websites.splitlines() if u.strip()]
        else:
            websites = [str(u).strip() for u in (websites or []) if str(u).strip()]

        def fetch_docs(urls):
            docs = BeautifulSoupWebReader().load_data(urls) or []

            for document in docs:
                if hasattr(document, "metadata") and isinstance(getattr(document, "metadata"), dict):
                    document.metadata = sanitize_for_json(document.metadata)
                if hasattr(document, "extra_info") and isinstance(getattr(document, "extra_info"), dict):
                    document.extra_info = sanitize_for_json(document.extra_info)

            valid = []
            for document in docs:
                if document is None:
                    continue

                text = getattr(document, "text", None)
                if text is None and hasattr(document, "get_content"):
                    try:
                        text = document.get_content()
                    except Exception:
                        text = None

                if text is None or str(text).strip() == "":
                    continue

                valid.append(document)

            return valid

        documents = fetch_docs(websites)
        if not documents:
            fallback_websites = [f"https://r.jina.ai/{u}" for u in websites]
            documents = fetch_docs(fallback_websites)

        if not documents:
            raise ValueError("No extractable text from the given URL(s).")

        pipeline = AdvancedIngestionPipeline()
        pipeline.disable_cache = True
        pipeline.cache = None
        nodes = pipeline.run(documents=documents) or []
        if not nodes:
            return []

        self.insert_nodes(nodes)
        return nodes

    def delete_ref_doc(self, ref_doc_id):
        self.index.delete_ref_doc(ref_doc_id=ref_doc_id, delete_from_docstore=True)
        self.storage_context.persist(persist_dir=self.persist_dir)
        print("Deleted document", ref_doc_id)

    def reset_index_storage(self):
        if not DEV_MODE:
            raise RuntimeError("Full rebuild storage reset is only implemented for development file storage.")

        storage_dir = Path(STORAGE_DIR)
        if storage_dir.exists():
            backup_dir = storage_dir / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
            copied = False
            for file_name in INDEX_STORAGE_FILES:
                source = storage_dir / file_name
                if source.exists():
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, backup_dir / file_name)
                    source.unlink()
                    copied = True
            if copied:
                print(f"Backed up previous index storage to {backup_dir}")

        self._set_storage_context(StorageContext.from_defaults(), STORAGE_DIR)
        print("Reset development index storage context")

    def _ingest_parsed_files(self, parsed):
        indexable_documents, _ = prepare_documents_for_indexing(parsed.native_documents)
        if not indexable_documents and not parsed.api_nodes:
            print("No indexable document text found")
            return {"nodes": [], "quality_report": parsed.quality_report}

        nodes = []
        if indexable_documents:
            pipeline = AdvancedIngestionPipeline(split_text=True)
            nodes.extend(pipeline.run(documents=indexable_documents) or [])
        if parsed.api_nodes:
            pipeline = AdvancedIngestionPipeline(split_text=False)
            nodes.extend(pipeline.run(nodes=parsed.api_nodes) or [])
        if nodes:
            self.insert_nodes(nodes)
        return {"nodes": nodes, "quality_report": parsed.quality_report}

    def _rebuild_dir_with_staging(self, input_dir, chunk_size, chunk_overlap):
        old_context = self.storage_context
        old_index = self.index
        old_index_id = self.index_id
        old_persist_dir = self.persist_dir

        self._cleanup_staging_storage()
        self._set_storage_context(StorageContext.from_defaults(), STAGING_STORAGE_DIR)
        try:
            result = self.load_dir(input_dir, chunk_size, chunk_overlap)
            if not result.get("nodes"):
                self._set_storage_context(old_context, old_persist_dir)
                self.index = old_index
                self.index_id = old_index_id
                self._cleanup_staging_storage()
                return result

            self._commit_staged_storage(Path(STAGING_STORAGE_DIR))
            self._reload_storage_context()
            return result
        except Exception:
            self._set_storage_context(old_context, old_persist_dir)
            self.index = old_index
            self.index_id = old_index_id
            self._cleanup_staging_storage()
            raise

    def _commit_staged_storage(self, staging_dir: Path):
        storage_dir = Path(STORAGE_DIR)
        storage_dir.mkdir(parents=True, exist_ok=True)
        staged_files = [staging_dir / name for name in INDEX_STORAGE_FILES if (staging_dir / name).exists()]
        if not staged_files:
            raise RuntimeError("Staged index storage was not generated.")

        backup_dir = storage_dir / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        copied = False
        for file_name in INDEX_STORAGE_FILES:
            source = storage_dir / file_name
            if source.exists():
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, backup_dir / file_name)
                source.unlink()
                copied = True
        if copied:
            print(f"Backed up previous index storage to {backup_dir}")

        for staged_file in staged_files:
            shutil.move(str(staged_file), str(storage_dir / staged_file.name))
        self._cleanup_staging_storage()

    def _cleanup_staging_storage(self):
        staging_dir = Path(STAGING_STORAGE_DIR)
        if not staging_dir.exists():
            return
        cwd = Path.cwd().resolve()
        resolved = staging_dir.resolve()
        if cwd not in resolved.parents and resolved != cwd:
            raise RuntimeError(f"Refusing to delete staging directory outside workspace: {resolved}")
        shutil.rmtree(staging_dir)

    def _reload_storage_context(self):
        context = StorageContext.from_defaults(persist_dir="./" + STORAGE_DIR)
        self._set_storage_context(context, STORAGE_DIR)

    def _set_storage_context(self, storage_context, persist_dir):
        import server.ingestion as ingestion_module
        import server.stores.strage_context as storage_module

        self.storage_context = storage_context
        self.persist_dir = persist_dir
        self.index = None
        self.index_id = None

        global STORAGE_CONTEXT
        STORAGE_CONTEXT = self.storage_context
        ingestion_module.STORAGE_CONTEXT = self.storage_context
        storage_module.STORAGE_CONTEXT = self.storage_context
