"""
Deduplication & Cache — STEP 0 of the pipeline.

Computes SHA-256 of the uploaded file bytes and queries MongoDB to check
whether this exact document has been processed before.

On cache hit:  returns (existing_doc_id, cached_doc_dict)
On cache miss: returns (None, None) — caller proceeds with full pipeline
"""

import hashlib
from typing import Optional, Tuple


def compute_sha256(file_bytes: bytes) -> str:
    """Return hex-encoded SHA-256 of raw file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def lookup_cache(
    file_bytes: bytes,
    doc_store,          # MongoDocStore instance
) -> Tuple[Optional[str], Optional[dict]]:
    """
    Check whether this file has already been processed.

    Returns:
        (doc_id, doc_dict)  — if cache hit
        (None,   None)      — if cache miss
    """
    file_hash = compute_sha256(file_bytes)

    # MongoDocStore uses doc_id as _id; hashes are stored separately
    # under the key "hash:{sha256}" → doc_id
    pointer_key = f"hash:{file_hash}"
    pointer = doc_store.get(pointer_key)

    if pointer is None:
        return None, None

    # pointer is a tiny dict {"doc_id": "..."}
    cached_doc_id = pointer.get("doc_id")
    if not cached_doc_id:
        return None, None

    cached_doc = doc_store.get(cached_doc_id)
    if cached_doc is None:
        # Stale pointer — hash record exists but document was evicted
        return None, None

    print(f"[Cache] ✅ Cache hit for sha256={file_hash[:16]}... → doc_id={cached_doc_id}")
    return cached_doc_id, cached_doc


def register_cache(
    file_bytes: bytes,
    doc_id: str,
    doc_store,
) -> None:
    """
    After a successful extraction, record sha256 → doc_id so future
    uploads of the same file skip MinerU entirely.
    """
    file_hash = compute_sha256(file_bytes)
    pointer_key = f"hash:{file_hash}"
    doc_store.set(pointer_key, {"doc_id": doc_id, "sha256": file_hash})
    print(f"[Cache] 📌 Registered sha256={file_hash[:16]}... → doc_id={doc_id}")
