"""
RAG Pipeline Service Module (RAG-11, RAG-13, RAG-14, RAG-16, Scalable Cloud-Native)
Orchestrates end-to-end RAG flow:
PDF Ingestion -> Chunking -> Embeddings -> Vector Search -> Context Injection -> LLM Answer Generation -> Citation.
Features:
- In-memory answer caching with TTL.
- Structured per-stage latency tracking (embed, retrieve, generate).
- In-memory PDF upload processing with size and page limits (no disk persistence).
- Idempotent content hashing (SHA-256).
- Strict context token budgeting to prevent free-tier quota exhaustion.
- Anti-hallucination safe refusal policy.
"""

import os
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from document_loader import load_documents_from_directory, load_pdf_with_metadata, load_pdf_from_bytes
from chunker import chunk_document_pages
from embeddings import BaseEmbedder, get_embedder
from vector_store import BaseVectorStore, get_vector_store
from retriever import SemanticRetriever
from llm_service import BaseLLMService, get_llm_service, UNKNOWN_ANSWER_PHRASE


class RAGService:
    """
    Complete Retrieval-Augmented Generation Service supporting Cloud Providers, Caching, and Observability.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        embed_provider: Optional[str] = None,
        vector_provider: Optional[str] = None,
        vector_store_dir: str = "./chroma_db",
        top_k: int = 4,
        cache_ttl_seconds: int = 600
    ):
        self.embedder: BaseEmbedder = get_embedder(provider=embed_provider)
        embed_dim = getattr(self.embedder, "dimension", 384)
        self.vector_store: BaseVectorStore = get_vector_store(
            provider=vector_provider,
            persist_directory=vector_store_dir,
            embedding_dimension=embed_dim
        )
        self.retriever = SemanticRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder
        )
        self.llm_service: BaseLLMService = get_llm_service(provider=llm_provider)
        self.top_k = top_k
        self.cache_ttl = cache_ttl_seconds

        # In-memory Answer Cache: {hash: (timestamp, result_dict)}
        self._answer_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

        # Structured Observability Metrics
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_retrieve_ms": 0.0,
            "avg_generate_ms": 0.0,
            "avg_total_ms": 0.0
        }

    def check_health(self) -> Dict[str, Any]:
        """Validates connection status across cloud and local services."""
        llm_ok = self.llm_service.is_available()
        embed_ok = self.embedder.is_available()
        stats = self.vector_store.get_stats()

        llm_type = self.llm_service.__class__.__name__
        embed_type = self.embedder.__class__.__name__

        has_error = bool(stats.get("error"))
        return {
            "llm_service": llm_type,
            "llm_online": llm_ok,
            "llm_model": getattr(self.llm_service, "model_name", "Mock"),
            "embedding_service": embed_type,
            "embedding_online": embed_ok,
            "vector_store_provider": stats.get("provider", "ChromaDB"),
            "indexed_chunks": stats.get("total_chunks", 0),
            "indexed_documents": stats.get("unique_documents", []),
            "all_healthy": bool(llm_ok and embed_ok and not has_error)
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Returns structured observability and latency metrics."""
        return {
            **self.metrics,
            "cache_size": len(self._answer_cache),
            "cache_hit_rate": round(
                (self.metrics["cache_hits"] / max(1, self.metrics["total_queries"])) * 100, 1
            )
        }

    def index_directory(
        self,
        dir_path: str = "./data",
        chunk_size: int = 1500,
        overlap: int = 200,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Indexes all PDF files in the target directory into the vector store.
        Safely clears collection if force_reindex=True.
        """
        if not os.path.exists(dir_path):
            return {"status": "error", "message": f"Directory not found: {dir_path}", "indexed_chunks": 0}

        if force_reindex:
            self.reset_index()

        pages_data, summaries = load_documents_from_directory(dir_path)
        if not pages_data:
            return {
                "status": "empty",
                "message": "No readable PDF pages found in directory.",
                "indexed_chunks": 0,
                "summaries": summaries
            }

        chunks = chunk_document_pages(pages_data, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return {
                "status": "empty",
                "message": "No text chunks generated from pages.",
                "indexed_chunks": 0,
                "summaries": summaries
            }

        # Add content hash to chunk metadata for idempotent tracking
        for c in chunks:
            c["content_hash"] = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()

        try:
            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.get_embeddings_batch(texts)
            count = self.vector_store.add_chunks(chunks, embeddings)

            scanned_docs = [s["source"] for s in summaries if s.get("is_scanned")]
            error_docs = [s["source"] for s in summaries if s.get("error")]

            return {
                "status": "success",
                "indexed_chunks": count,
                "total_pages": len(pages_data),
                "documents": list({p["source"] for p in pages_data}),
                "scanned_documents": scanned_docs,
                "failed_documents": error_docs,
                "summaries": summaries
            }
        except Exception as e:
            return {"status": "error", "message": f"Embedding or storage failure: {e}", "indexed_chunks": 0}

    def index_pdf_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_size: int = 1500,
        overlap: int = 200,
        max_size_mb: int = 15,
        max_pages: int = 50
    ) -> Dict[str, Any]:
        """
        Indexes a PDF directly from in-memory bytes without persisting raw PDF to disk.
        Enforces size and page budget guards to protect container RAM (1GB).
        Safely clears existing chunks for the same filename to avoid duplication.
        All text chunks, hashes, and source metadata are stored in vector database payloads.
        """
        # Guard 1: File size limit
        if len(file_bytes) > max_size_mb * 1024 * 1024:
            return {
                "status": "error",
                "message": f"Upload exceeds maximum allowable size of {max_size_mb} MB.",
                "filename": filename,
                "indexed_chunks": 0
            }

        pages_data, summary = load_pdf_from_bytes(file_bytes, filename)

        if summary.get("error"):
            return {
                "status": "error",
                "message": summary["error"],
                "filename": filename,
                "indexed_chunks": 0
            }

        # Guard 2: Page count limit
        if summary.get("total_pages", 0) > max_pages:
            return {
                "status": "error",
                "message": f"Upload exceeds maximum allowable page limit ({summary.get('total_pages')} > {max_pages} pages).",
                "filename": filename,
                "indexed_chunks": 0
            }

        if summary.get("is_scanned"):
            return {
                "status": "warning",
                "message": "Document contains no extractable text. Scanned image PDFs require an OCR engine.",
                "filename": filename,
                "indexed_chunks": 0
            }

        chunks = chunk_document_pages(pages_data, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return {
                "status": "empty",
                "message": "Document produced no text chunks.",
                "filename": filename,
                "indexed_chunks": 0
            }

        # Guard 3: Add SHA-256 content hash
        for c in chunks:
            c["content_hash"] = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()

        try:
            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.get_embeddings_batch(texts)
            count = self.vector_store.safe_reindex_document(chunks, embeddings)
            # Invalidate answer cache since vector store updated
            self._answer_cache.clear()
            return {
                "status": "success",
                "indexed_chunks": count,
                "filename": filename
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Embedding or storage failure: {e}",
                "filename": filename,
                "indexed_chunks": 0
            }

    def format_context(self, retrieved_chunks: List[Dict[str, Any]], max_chars: int = 12000) -> str:
        """
        Formats retrieved chunks into contextual blocks for the LLM prompt.
        Enforces token budget limit (max_chars ~3,000 tokens) to protect free-tier rate limits.
        """
        context_blocks = []
        accumulated_chars = 0

        for chunk in retrieved_chunks:
            source = chunk.get("source", "Unknown Document")
            page = chunk.get("page", 1)
            text = chunk.get("text", "").strip()
            block = f"[Document: {source} | Page: {page}]\n{text}"

            if accumulated_chars + len(block) > max_chars:
                break

            context_blocks.append(block)
            accumulated_chars += len(block)

        return "\n\n".join(context_blocks)

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Complete question answering workflow:
        1. Check in-memory answer cache.
        2. Retrieve top-k chunks with latency tracking.
        3. Enforce context token budget.
        4. Synthesize answer via configured LLM (with fallback).
        5. Enforce anti-hallucination refusal and format citations.
        """
        start_time = time.time()
        k = top_k or self.top_k
        clean_question = question.strip() if question else ""

        if not clean_question:
            return {
                "answer": "Please enter a valid question.",
                "sources": [],
                "retrieved_chunks": [],
                "is_refusal": False,
                "is_error": False,
                "latency_ms": 0.0
            }

        # Check Cache
        cache_key = hashlib.md5(f"{clean_question.lower()}_{k}".encode("utf-8")).hexdigest()
        now = time.time()
        self.metrics["total_queries"] += 1

        if use_cache and cache_key in self._answer_cache:
            cached_time, cached_result = self._answer_cache[cache_key]
            if now - cached_time < self.cache_ttl:
                self.metrics["cache_hits"] += 1
                result_copy = dict(cached_result)
                result_copy["from_cache"] = True
                result_copy["latency_ms"] = round((time.time() - start_time) * 1000, 1)
                return result_copy

        self.metrics["cache_misses"] += 1

        # 1. Retrieve relevant chunks
        t_ret_start = time.time()
        try:
            retrieved_chunks = self.retriever.retrieve(clean_question, top_k=k)
        except Exception as e:
            return {
                "answer": f"Retrieval error: {e}",
                "sources": [],
                "retrieved_chunks": [],
                "is_refusal": False,
                "is_error": True,
                "latency_ms": round((time.time() - start_time) * 1000, 1)
            }
        ret_latency_ms = round((time.time() - t_ret_start) * 1000, 1)

        if not retrieved_chunks:
            return {
                "answer": UNKNOWN_ANSWER_PHRASE,
                "sources": [],
                "retrieved_chunks": [],
                "is_refusal": True,
                "is_error": False,
                "latency_ms": round((time.time() - start_time) * 1000, 1)
            }

        # 2. Build context with token budget
        context = self.format_context(retrieved_chunks)

        # 3. Construct prompt
        prompt = self.llm_service.build_prompt(clean_question, context)

        # 4. Generate answer from configured LLM
        t_gen_start = time.time()
        try:
            raw_answer = self.llm_service.generate_response(prompt)
        except Exception as e:
            err_msg = str(e)
            return {
                "answer": f"⚠️ {err_msg}",
                "sources": [],
                "retrieved_chunks": retrieved_chunks,
                "is_refusal": False,
                "is_error": True,
                "latency_ms": round((time.time() - start_time) * 1000, 1)
            }
        gen_latency_ms = round((time.time() - t_gen_start) * 1000, 1)

        # 5. Check if LLM indicates answer not found
        is_refusal = (
            UNKNOWN_ANSWER_PHRASE.lower() in raw_answer.lower()
            or "not found in the available documents" in raw_answer.lower()
            or "information not found" in raw_answer.lower()
        )

        # 6. Extract unique sources
        sources = list(dict.fromkeys(c.get("source") for c in retrieved_chunks if c.get("source")))
        total_latency_ms = round((time.time() - start_time) * 1000, 1)

        # Update metrics averages
        q_count = self.metrics["total_queries"]
        self.metrics["avg_retrieve_ms"] = round(
            (self.metrics["avg_retrieve_ms"] * (q_count - 1) + ret_latency_ms) / q_count, 1
        )
        self.metrics["avg_generate_ms"] = round(
            (self.metrics["avg_generate_ms"] * (q_count - 1) + gen_latency_ms) / q_count, 1
        )
        self.metrics["avg_total_ms"] = round(
            (self.metrics["avg_total_ms"] * (q_count - 1) + total_latency_ms) / q_count, 1
        )

        res = {
            "answer": raw_answer,
            "sources": [] if is_refusal else sources,
            "retrieved_chunks": retrieved_chunks,
            "is_refusal": is_refusal,
            "is_error": False,
            "prompt_used": prompt,
            "from_cache": False,
            "latency_ms": total_latency_ms,
            "retrieve_ms": ret_latency_ms,
            "generate_ms": gen_latency_ms
        }

        # Cache valid answers
        if not res["is_error"]:
            self._answer_cache[cache_key] = (now, res)

        return res

    def get_stats(self) -> Dict[str, Any]:
        return self.vector_store.get_stats()

    def reset_index(self):
        self._answer_cache.clear()
        self.vector_store.reset_collection()
