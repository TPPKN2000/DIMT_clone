"""
MongoDB persistence layer for doc_store.

Mirrors the in-memory doc_store dict but persists to MongoDB.
Uses MONGODB_URL from .env. Falls back gracefully if MongoDB is unavailable.
"""

import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()


class MongoDocStore:
    """Dual-write store: in-memory dict + MongoDB collection."""

    def __init__(self):
        self._cache: dict = {}
        self._db = None
        self._collection = None
        self._connect()

    def _connect(self):
        url = os.environ.get("MONGODB_URL")
        if not url:
            print("[MongoDB] No MONGODB_URL in .env — running in-memory only")
            return
        try:
            from pymongo import MongoClient
            from pymongo.server_api import ServerApi
            client = MongoClient(url, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
            # Test connection
            client.admin.command("ping")
            self._db = client["dimt-data"]
            self._collection = self._db["documents"]
            print("[MongoDB] Connected successfully to dimt-data")
        except Exception as e:
            print(f"[MongoDB] Connection failed (in-memory fallback): {e}")

    def _serialize_for_mongo(self, doc: dict) -> dict:
        """Prepare doc dict for MongoDB storage (skip non-serializable fields)."""
        safe = {}
        skip_keys = {"translated_middle", "middle_json"}  # large JSON — store as string
        for k, v in doc.items():
            if k in skip_keys and isinstance(v, dict):
                safe[k] = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, (str, int, float, bool, type(None))):
                safe[k] = v
            elif isinstance(v, dict):
                safe[k] = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, list):
                safe[k] = json.dumps(v, ensure_ascii=False)
            else:
                safe[k] = str(v)
        return safe

    def _deserialize_from_mongo(self, mongo_doc: dict) -> dict:
        """Restore doc dict from MongoDB."""
        if not mongo_doc:
            return {}
        doc = {}
        json_keys = {"translated_middle", "middle_json", "agent_result"}
        for k, v in mongo_doc.items():
            if k == "_id":
                continue
            if k in json_keys and isinstance(v, str):
                try:
                    doc[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    doc[k] = v
            else:
                doc[k] = v
        return doc

    def set(self, doc_id: str, data: dict):
        """Store document data (both in-memory and MongoDB)."""
        self._cache[doc_id] = data
        if self._collection is not None:
            try:
                safe = self._serialize_for_mongo(data)
                safe["_id"] = doc_id
                safe["updated_at"] = datetime.now(timezone.utc)
                self._collection.replace_one(
                    {"_id": doc_id}, safe, upsert=True
                )
            except Exception as e:
                print(f"[MongoDB] Write error for {doc_id}: {e}")

    def get(self, doc_id: str) -> dict | None:
        """Get document data (in-memory first, MongoDB fallback)."""
        if doc_id in self._cache:
            return self._cache[doc_id]
        if self._collection is not None:
            try:
                mongo_doc = self._collection.find_one({"_id": doc_id})
                if mongo_doc:
                    doc = self._deserialize_from_mongo(mongo_doc)
                    self._cache[doc_id] = doc
                    return doc
            except Exception as e:
                print(f"[MongoDB] Read error for {doc_id}: {e}")
        return None

    def update(self, doc_id: str, updates: dict):
        """Update specific fields of a document."""
        doc = self.get(doc_id)
        if doc is None:
            return
        doc.update(updates)
        self.set(doc_id, doc)
