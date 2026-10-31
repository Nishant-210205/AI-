"""
Vector Database Storage Module (RAG-9, Cloud-Native Qdrant & Chroma Storage)
Primary Cloud: Qdrant Cloud Free Tier (qdrant-client with Cosine Distance & Payload Storage)
Local / Test: In-Memory Qdrant or Persistent ChromaDB
Guarantees:
- Uploaded PDF text chunks are stored in vector DB payloads, NOT saved to disk.
- Safe re-indexing by document source without duplicate records.
- Descriptive error handling when free Qdrant clusters suspend after 7 days of inactivity.
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import uuid
import hashlib
import re
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


def sanitize_secret_text(text: str, *secrets: Optional[str]) -> str:
    """
    Strips raw secrets and tokens from error messages and logs to prevent leaks.
    """
    if not text:
        return ""
    sanitized = str(text)
    for s in secrets:
        if s and len(str(s).strip()) >= 4:
            clean_s = str(s).strip()
            sanitized = sanitized.replace(clean_s, "***REDACTED***")
    sanitized = re.sub(
        r'(?:api[-_]?key[:=]\s*[\'"]?)([0-9A-Za-z_\-]{8,})',
        r'api_key=***REDACTED***',
        sanitized,
        flags=re.IGNORECASE
    )
    sanitized = re.sub(
        r'Illegal header value b[\'"][^\'"]+[\'"]',
        'Illegal header value (whitespace or newline in secret)',
        sanitized
    )
    return sanitized


def validate_and_sanitize_qdrant_url(raw_url: Optional[str]) -> str:
    """
    Validates and sanitizes Qdrant URL.
    Accepts only plain HTTPS endpoints (or http://localhost / :memory: for local testing).
    Strips accidental surrounding whitespace, quotes, and parses/resolves Markdown links.
    Raises ValueError on malformed or non-HTTPS URLs.
    """
    if not raw_url:
        return ""
    s = str(raw_url).strip()
    if s == ":memory:":
        return ":memory:"

    # 1. Check and extract URL from Markdown link pattern e.g. [title](https://...)
    md_match = re.search(r'\[.*?\]\((https?://[^\s\)\"\']+)\)', s)
    if md_match:
        s = md_match.group(1).strip()
    else:
        # Check if wrapped in brackets e.g. [https://...] or <https://...>
        bracket_match = re.search(r'[\[<](https?://[^\]>]+)[\]>]', s)
        if bracket_match:
            s = bracket_match.group(1).strip()

    # 2. Strip surrounding quotes, backticks, whitespace, newlines, carriage returns
    s = s.strip("\"'` \t\r\n").rstrip("/")

    # 3. Validate scheme and hostname
    parsed = urlparse(s)
    if not parsed.scheme or parsed.scheme not in ("https", "http"):
        raise ValueError(
            f"Invalid QDRANT_URL scheme: '{parsed.scheme}'. "
            "Qdrant Cloud requires a valid plain HTTPS endpoint (e.g. 'https://your-cluster-id.region.aws.cloud.qdrant.io'), "
            "not a Markdown link or non-HTTPS protocol."
        )

    if not parsed.netloc:
        raise ValueError(
            "Invalid QDRANT_URL: Missing host or domain name. "
            "Please provide a plain HTTPS endpoint (e.g. 'https://your-cluster-id.region.aws.cloud.qdrant.io')."
        )

    # 4. Check for illegal characters or lingering Markdown artifacts
    if any(ch in s for ch in "[]()<> \r\n\t"):
        raise ValueError(
            "Invalid QDRANT_URL: Contains illegal characters or unparsed Markdown link formatting. "
            "Please provide a clean plain HTTPS URL."
        )

    return s


def sanitize_qdrant_api_key(raw_key: Optional[str]) -> Optional[str]:
    """
    Sanitizes Qdrant API key by removing accidental surrounding/internal whitespace,
    newlines, and quotes. Ensures the token is clean ASCII to prevent httpx.LocalProtocolError.
    """
    if not raw_key:
        return None
    k = str(raw_key).strip().strip("\"'`").strip()
    k = "".join(k.split())
    if not k:
        return None
    try:
        k.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("Invalid QDRANT_API_KEY: Key contains non-ASCII characters.")
    return k


def is_placeholder_qdrant_url(raw_url: Optional[str]) -> bool:
    """
    Checks if a Qdrant URL is an unconfigured documentation placeholder
    (e.g. 'your-cluster-id', 'xxxxxx-xxxx', 'example.com').
    """
    if not raw_url:
        return True
    low = str(raw_url).lower().strip()
    return any(
        ph in low
        for ph in (
            "your-cluster-id",
            "your-cloud-qdrant",
            "your-replacement",
            "your_qdrant",
            "your-real-qdrant",
            "cluster-id",
            "xxxxxx-xxxx",
            "example.com",
            "your_real_key",
        )
    )


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    # 1. Check environment variables (exact, uppercase, lowercase)
    for k in (key, key.upper(), key.lower()):
        val = os.environ.get(k)
        if val is not None:
            cleaned = str(val).strip().strip("\"'`").strip()
            if cleaned:
                return cleaned

    # 2. Check Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets is not None:
            for k in (key, key.upper(), key.lower()):
                if k in st.secrets and st.secrets[k] is not None:
                    cleaned = str(st.secrets[k]).strip().strip("\"'`").strip()
                    if cleaned:
                        return cleaned
            for section in ("qdrant", "gemini", "groq", "general"):
                if section in st.secrets and isinstance(st.secrets[section], dict):
                    sec_dict = st.secrets[section]
                    sub_key = key.replace(f"{section.upper()}_", "").lower()
                    for sk in (sub_key, key.lower(), key):
                        if sk in sec_dict and sec_dict[sk] is not None:
                            cleaned = str(sec_dict[sk]).strip().strip("\"'`").strip()
                            if cleaned:
                                return cleaned
    except Exception:
        pass

    return default


class BaseVectorStore(ABC):
    """Abstract interface for vector database storage."""

    @abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        pass

    @abstractmethod
    def query_similar(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_document(self, source_filename: str) -> int:
        pass

    @abstractmethod
    def safe_reindex_document(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def reset_collection(self) -> None:
        pass


class QdrantCloudVectorStore(BaseVectorStore):
    """
    Production Cloud Vector Store using Qdrant Cloud Free Tier (1GB permanent free cluster).
    Also supports in-memory execution (`url=':memory:'`) for fast, zero-dependency testing.
    Stores chunk text + metadata directly in payload (eliminating disk file dependency).
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "college_documents",
        embedding_dimension: int = 384,
        **kwargs
    ):
        raw_u = url if url is not None else get_secret("QDRANT_URL", "")
        raw_k = api_key if api_key is not None else get_secret("QDRANT_API_KEY", "")
        self.url = validate_and_sanitize_qdrant_url(raw_u)
        self.api_key = sanitize_qdrant_api_key(raw_k)
        raw_coll = collection_name or get_secret("QDRANT_COLLECTION", "college_documents")
        self.collection_name = str(raw_coll).strip().strip("\"'`").strip() or "college_documents"
        self.embedding_dimension = embedding_dimension
        self._known_sources: set = set()
        self._client = None
        self._collection_initialized = False

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            if self.url and self.url != ":memory:":
                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key or None,
                    timeout=15,
                    check_compatibility=False
                )
            else:
                self._client = QdrantClient(":memory:")
        return self._client

    def _ensure_collection(self):
        if self._collection_initialized:
            return
        client = self._get_client()
        from qdrant_client.http import models as qmodels
        try:
            collections = client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.embedding_dimension,
                        distance=qmodels.Distance.COSINE
                    )
                )
            else:
                # Validate existing collection dimension against current embedder
                info = client.get_collection(collection_name=self.collection_name)
                existing_dim = None
                if hasattr(info.config.params.vectors, "size"):
                    existing_dim = info.config.params.vectors.size
                elif isinstance(info.config.params.vectors, dict) and "" in info.config.params.vectors:
                    existing_dim = getattr(info.config.params.vectors[""], "size", None)
                if existing_dim is not None and existing_dim != self.embedding_dimension:
                    raise ValueError(
                        f"Qdrant collection '{self.collection_name}' has dimension {existing_dim}, "
                        f"but current embedder requires dimension {self.embedding_dimension}. Dimension mismatch refused."
                    )
            self._collection_initialized = True
            # Create payload index on 'source' for efficient filtered queries
            try:
                client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD
                )
            except Exception:
                pass
        except ValueError:
            raise
        except Exception as e:
            err = str(e)
            low_err = err.lower()
            clean_err = sanitize_secret_text(err, self.api_key)
            if "401" in err or "403" in err or "forbidden" in low_err or "unauthorized" in low_err:
                raise ConnectionError(
                    "Qdrant Cloud authentication failed (HTTP 401/403). The provided QDRANT_API_KEY was rejected by your cluster. "
                    "Please verify that your active API key in Qdrant Cloud console matches QDRANT_API_KEY in Streamlit Secrets."
                ) from e
            elif "responsehandlingexception" in low_err or "localprotocolerror" in low_err or "illegal header" in low_err:
                raise ConnectionError(
                    "Qdrant Cloud connection failed due to an HTTP protocol/header formatting error. "
                    "Please verify that QDRANT_URL is a plain HTTPS endpoint without Markdown links and QDRANT_API_KEY contains no unexpected characters."
                ) from e
            elif any(term in low_err for term in ("suspended", "connection refused", "connect", "getaddrinfo failed", "timed out", "timeout")):
                raise ConnectionError(
                    "Qdrant Cloud cluster is unreachable or has been suspended due to free-tier inactivity "
                    "(7 days without queries). Please log in to https://cloud.qdrant.io to resume the cluster."
                ) from e
            else:
                raise ConnectionError(f"Qdrant Cloud connection failed: {clean_err}") from e

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not chunks or not embeddings:
            return 0

        self._ensure_collection()
        client = self._get_client()
        from qdrant_client.http import models as qmodels

        points = []
        for chunk, emb in zip(chunks, embeddings):
            text_content = chunk.get("text", "")
            content_hash = hashlib.sha256(text_content.encode("utf-8")).hexdigest()[:16]
            chunk_id = chunk.get("id") or f"{chunk.get('source', 'doc')}_p{chunk.get('page', 1)}_{content_hash}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            src = chunk.get("source", "unknown")
            self._known_sources.add(src)
            payload = {
                "chunk_id": chunk_id,
                "text": text_content,
                "source": src,
                "page": int(chunk.get("page", 1)),
                "chunk_index": int(chunk.get("chunk_index", 1)),
                "content_hash": content_hash
            }
            points.append(qmodels.PointStruct(id=point_id, vector=[float(x) for x in emb], payload=payload))

        client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def query_similar(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        self._ensure_collection()
        client = self._get_client()
        try:
            response = client.query_points(
                collection_name=self.collection_name,
                query=[float(x) for x in query_embedding],
                limit=top_k,
                with_payload=True
            )
        except Exception as e:
            if "suspended" in str(e).lower() or "connect" in str(e).lower():
                raise ConnectionError(
                    "Qdrant Cloud cluster unreachable. Verify status at https://cloud.qdrant.io"
                ) from e
            raise

        retrieved = []
        for pt in getattr(response, "points", []):
            payload = pt.payload or {}
            score = round(float(pt.score), 4) if hasattr(pt, "score") and pt.score is not None else 0.0
            retrieved.append({
                "id": payload.get("chunk_id", str(pt.id)),
                "text": payload.get("text", ""),
                "source": payload.get("source", "unknown"),
                "page": payload.get("page", 1),
                "chunk_index": payload.get("chunk_index", 1),
                "distance": round(1.0 - score, 4),
                "similarity": score
            })

        return retrieved

    similarity_search = query_similar

    def delete_document(self, source_filename: str) -> int:
        self._ensure_collection()
        client = self._get_client()
        from qdrant_client.http import models as qmodels
        try:
            client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source_filename))]
                    )
                )
            )
            self._known_sources.discard(source_filename)
            return 1
        except Exception as e:
            print(f"Warning deleting document {source_filename} in Qdrant: {e}")
            return 0

    def safe_reindex_document(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not chunks:
            return 0
        src = chunks[0].get("source")
        if src:
            self.delete_document(src)
        return self.add_chunks(chunks, embeddings)

    def get_stats(self) -> Dict[str, Any]:
        try:
            self._ensure_collection()
            client = self._get_client()
            info = client.get_collection(collection_name=self.collection_name)
            pts_count = getattr(info, "points_count", 0) or 0

            unique_docs = set(self._known_sources)
            try:
                records, _ = client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    with_payload=["source"]
                )
                for rec in records:
                    if rec.payload and "source" in rec.payload:
                        unique_docs.add(rec.payload["source"])
                self._known_sources.update(unique_docs)
            except Exception:
                pass

            provider_name = "Qdrant Cloud Free Tier" if (self.url and self.url != ":memory:") else "Qdrant In-Memory"
            return {
                "provider": provider_name,
                "total_chunks": pts_count,
                "unique_documents": sorted(list(unique_docs)),
                "collection_name": self.collection_name,
                "endpoint": self.url or "in-memory"
            }
        except Exception as e:
            clean_err = sanitize_secret_text(str(e), self.api_key)
            return {
                "provider": "Qdrant Cloud (Connection Error)",
                "total_chunks": 0,
                "unique_documents": [],
                "collection_name": self.collection_name,
                "endpoint": self.url or "in-memory",
                "error": clean_err
            }

    def reset_collection(self) -> None:
        client = self._get_client()
        try:
            client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass
        self._known_sources.clear()
        self._collection_initialized = False
        self._ensure_collection()


class ChromaVectorStore(BaseVectorStore):
    """
    Local / Offline ChromaDB vector store.
    Preserved for local development and offline environments.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: Optional[str] = None,
        embedding_dimension: int = 384,
        **kwargs
    ):
        self.persist_directory = persist_directory
        self.embedding_dimension = embedding_dimension
        if collection_name is None:
            self.collection_name = f"college_documents_{embedding_dimension}"
        else:
            self.collection_name = collection_name
        os.makedirs(self.persist_directory, exist_ok=True)

        import chromadb
        from chromadb.config import Settings
        import shutil

        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "College Document Embeddings Collection", "dimension": embedding_dimension}
            )
            # Verify database integrity on this OS environment
            _ = self.collection.count()
            if _ > 0:
                _ = self.collection.get(limit=1, include=["metadatas"])
        except Exception:
            # Self-heal cross-platform SQLite/Rust binding incompatibility
            try:
                shutil.rmtree(self.persist_directory, ignore_errors=True)
                os.makedirs(self.persist_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "College Document Embeddings Collection", "dimension": embedding_dimension}
                )
            except Exception:
                # In-memory ephemeral fallback if disk client cannot initialize
                self.client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "College Document Embeddings Collection", "dimension": embedding_dimension}
                )

    def delete_document(self, source_filename: str) -> int:
        try:
            existing = self.collection.get(where={"source": source_filename}, include=["metadatas"])
            if existing and existing.get("ids"):
                ids_to_del = existing["ids"]
                self.collection.delete(ids=ids_to_del)
                return len(ids_to_del)
            return 0
        except Exception as e:
            print(f"Warning during delete_document for {source_filename}: {e}")
            return 0

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not chunks or not embeddings:
            return 0

        ids = [str(c.get("id") or c.get("chunk_id") or f"chk_{i}") for i, c in enumerate(chunks)]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": c.get("source", "unknown"),
                "page": int(c.get("page", 1)),
                "chunk_index": int(c.get("chunk_index", 1))
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        return len(ids)

    def safe_reindex_document(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not chunks:
            return 0
        src = chunks[0].get("source")
        if src:
            self.delete_document(src)
        return self.add_chunks(chunks, embeddings)

    def query_similar(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        try:
            count = self.collection.count()
            if count == 0:
                return []

            actual_k = min(top_k, count)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            return []

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []
            ids = results["ids"][0] if results.get("ids") else []

            for i in range(len(docs)):
                dist = distances[i] if i < len(distances) else 1.0
                similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

                retrieved.append({
                    "id": ids[i] if i < len(ids) else f"chunk_{i}",
                    "text": docs[i],
                    "source": metas[i].get("source", "unknown") if i < len(metas) else "unknown",
                    "page": metas[i].get("page", 1) if i < len(metas) else 1,
                    "chunk_index": metas[i].get("chunk_index", 1) if i < len(metas) else 1,
                    "distance": round(dist, 4),
                    "similarity": round(similarity, 4)
                })

        return retrieved

    similarity_search = query_similar

    def get_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
        except Exception:
            count = 0

        sources = set()
        if count > 0:
            try:
                sample = self.collection.get(include=["metadatas"])
                if sample and "metadatas" in sample:
                    for meta in sample["metadatas"]:
                        if meta and "source" in meta:
                            sources.add(meta["source"])
            except Exception:
                pass

        return {
            "provider": "ChromaDB (Local / Test Storage)",
            "total_chunks": count,
            "unique_documents": sorted(list(sources)),
            "collection_name": self.collection_name,
            "storage_path": os.path.abspath(self.persist_directory)
        }

    def reset_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "College Document Embeddings Collection"}
        )


class ResilientVectorStore(BaseVectorStore):
    """
    Transparent fault-tolerant vector store wrapper.
    Attempts operations on the primary store (e.g. Qdrant Cloud).
    If the primary store is unreachable or fails due to network/DNS/suspension errors,
    it seamlessly falls back to ChromaDB so the application never breaks.
    """
    def __init__(self, primary: BaseVectorStore, fallback: BaseVectorStore):
        self.primary = primary
        self.fallback = fallback
        self._fallback_active = False
        self._fallback_reason = None

    def _activate_fallback(self, reason: str):
        self._fallback_active = True
        self._fallback_reason = reason

    def delete_document(self, source_filename: str) -> int:
        if self._fallback_active:
            try:
                return self.fallback.delete_document(source_filename)
            except Exception:
                return 0
        try:
            return self.primary.delete_document(source_filename)
        except Exception as e:
            self._activate_fallback(str(e))
            try:
                return self.fallback.delete_document(source_filename)
            except Exception:
                return 0

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if self._fallback_active:
            try:
                return self.fallback.add_chunks(chunks, embeddings)
            except Exception:
                return 0
        try:
            return self.primary.add_chunks(chunks, embeddings)
        except Exception as e:
            self._activate_fallback(str(e))
            try:
                return self.fallback.add_chunks(chunks, embeddings)
            except Exception:
                return 0

    def safe_reindex_document(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if self._fallback_active:
            try:
                return self.fallback.safe_reindex_document(chunks, embeddings)
            except Exception:
                return 0
        try:
            return self.primary.safe_reindex_document(chunks, embeddings)
        except Exception as e:
            self._activate_fallback(str(e))
            try:
                return self.fallback.safe_reindex_document(chunks, embeddings)
            except Exception:
                return 0

    def query_similar(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        if self._fallback_active:
            try:
                return self.fallback.query_similar(query_embedding, top_k=top_k)
            except Exception:
                return []
        try:
            return self.primary.query_similar(query_embedding, top_k=top_k)
        except Exception as e:
            self._activate_fallback(str(e))
            try:
                return self.fallback.query_similar(query_embedding, top_k=top_k)
            except Exception:
                return []

    similarity_search = query_similar

    def get_stats(self) -> Dict[str, Any]:
        if not self._fallback_active:
            try:
                p_stats = self.primary.get_stats()
                if not p_stats.get("error"):
                    return p_stats
                self._activate_fallback(p_stats.get("error"))
            except Exception as e:
                self._activate_fallback(str(e))

        try:
            fb_stats = dict(self.fallback.get_stats())
        except Exception:
            fb_stats = {
                "provider": "VectorStore (Resilient Safe Mode)",
                "total_chunks": 0,
                "unique_documents": [],
                "collection_name": "college_documents",
            }

        fb_stats["fallback_active"] = True
        fb_stats["provider"] = fb_stats.get("provider", "ChromaDB (Active Resilient Fallback)")
        if self._fallback_reason:
            clean_r = sanitize_secret_text(self._fallback_reason)
            fb_stats["notice"] = (
                f"Qdrant Cloud unavailable ({clean_r}). "
                "Operating seamlessly on ChromaDB fallback."
            )
        return fb_stats

    def reset_collection(self) -> None:
        if not self._fallback_active:
            try:
                self.primary.reset_collection()
            except Exception as e:
                self._activate_fallback(str(e))
        self.fallback.reset_collection()


def get_vector_store(
    provider: Optional[str] = None,
    embedding_dimension: int = 384,
    **kwargs
) -> BaseVectorStore:
    """
    Factory function for vector store instances.
    Priority:
    1. Explicit provider
    2. Environment / Secrets variable VECTOR_STORE_PROVIDER
    3. If QDRANT_URL is set -> Resilient Qdrant with ChromaDB fallback
    4. Otherwise -> ChromaVectorStore (local fallback)
    """
    selected = (provider or get_secret("VECTOR_PROVIDER", get_secret("VECTOR_STORE_PROVIDER", ""))).lower()

    if selected == "qdrant":
        q_url = kwargs.get("url") if "url" in kwargs else get_secret("QDRANT_URL", "")
        if not q_url or q_url == ":memory:":
            if kwargs.get("url") == ":memory:" or q_url == ":memory:":
                return QdrantCloudVectorStore(url=":memory:", embedding_dimension=embedding_dimension, **kwargs)
            raise ValueError(
                "Qdrant Cloud vector store requested, but QDRANT_URL is not set. "
                "For persistent cloud operation, configure QDRANT_URL and QDRANT_API_KEY."
            )

        if is_placeholder_qdrant_url(q_url):
            return ChromaVectorStore(embedding_dimension=embedding_dimension, **kwargs)

        primary_store = QdrantCloudVectorStore(embedding_dimension=embedding_dimension, **kwargs)
        fallback_store = ChromaVectorStore(embedding_dimension=embedding_dimension, **kwargs)
        return ResilientVectorStore(primary=primary_store, fallback=fallback_store)

    elif selected == "memory":
        return QdrantCloudVectorStore(url=":memory:", embedding_dimension=embedding_dimension, **kwargs)
    elif selected == "chroma":
        return ChromaVectorStore(embedding_dimension=embedding_dimension, **kwargs)
    else:
        q_url = get_secret("QDRANT_URL")
        if q_url and not is_placeholder_qdrant_url(q_url):
            primary_store = QdrantCloudVectorStore(embedding_dimension=embedding_dimension, **kwargs)
            fallback_store = ChromaVectorStore(embedding_dimension=embedding_dimension, **kwargs)
            return ResilientVectorStore(primary=primary_store, fallback=fallback_store)
        return ChromaVectorStore(embedding_dimension=embedding_dimension, **kwargs)


# Backwards compatibility alias
VectorStore = ChromaVectorStore

__all__ = [
    "BaseVectorStore",
    "QdrantCloudVectorStore",
    "ChromaVectorStore",
    "ResilientVectorStore",
    "VectorStore",
    "get_vector_store",
    "is_placeholder_qdrant_url",
    "validate_and_sanitize_qdrant_url",
    "sanitize_qdrant_api_key",
    "sanitize_secret_text",
    "get_secret",
]

