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


_LAYOUT_FIELDS = {"middle_json", "translated_middle"}
_JSON_FIELDS = {"agent_result"}


class MongoDocStore:
    """Dual-write store: in-memory dict + MongoDB collection."""

    def __init__(self):
        self._cache: dict = {}
        self._users_cache: dict = {}
        self._db = None
        self._collection = None
        self._layouts_collection = None
        self._users_collection = None
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
            self._layouts_collection = self._db["layouts"]
            self._users_collection = self._db["users"]
            print("[MongoDB] Connected successfully to dimt-data")
        except Exception as e:
            print(f"[MongoDB] Connection failed (in-memory fallback): {e}")

    def _serialize_small(self, data: dict) -> dict:
        out = {}
        for k, v in data.items():
            if k in _LAYOUT_FIELDS:
                continue
            if k in _JSON_FIELDS:
                out[k] = json.dumps(v, ensure_ascii=False) if v is not None else None
            elif isinstance(v, (str, int, float, bool, type(None), bytes)):
                out[k] = v
            else:
                out[k] = json.dumps(v, ensure_ascii=False)
        return out

    def _deserialize_small(self, mongo_doc: dict) -> dict:
        if not mongo_doc:
            return {}
        out = {}
        for k, v in mongo_doc.items():
            if k == "_id":
                out["_id"] = str(v)
                continue
            if k in _JSON_FIELDS and isinstance(v, str):
                try:
                    out[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    out[k] = v
            else:
                out[k] = v
        return out

    def _serialize_layout(self, doc_id: str, data: dict) -> dict:
        out = {"_id": doc_id, "updated_at": datetime.now(timezone.utc)}
        for field in _LAYOUT_FIELDS:
            val = data.get(field)
            if val is not None:
                if isinstance(val, (dict, list)):
                    out[field] = json.dumps(val, ensure_ascii=False)
                else:
                    out[field] = val
            else:
                out[field] = None
        return out

    def _deserialize_layout(self, mongo_doc: dict) -> dict:
        if not mongo_doc:
            return {}
        out = {}
        for field in _LAYOUT_FIELDS:
            val = mongo_doc.get(field)
            if val is not None and isinstance(val, str):
                try:
                    out[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    out[field] = val
            else:
                out[field] = val
        return out

    def _deserialize_from_mongo(self, mongo_doc: dict) -> dict:
        """Compatibility fallback to parse layout keys if they are in metadata document."""
        if not mongo_doc:
            return {}
        doc = {}
        json_keys = {"translated_middle", "middle_json", "agent_result"}
        for k, v in mongo_doc.items():
            if k == "_id":
                doc["_id"] = str(v)
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
        data["updated_at"] = datetime.now(timezone.utc)
        self._cache[doc_id] = data
        
        save_to_db = data.get("save_to_db", True)
        if save_to_db and self._collection is not None:
            try:
                # 1. Write metadata to documents collection
                small = self._serialize_small(data)
                small["_id"] = doc_id
                self._collection.replace_one(
                    {"_id": doc_id}, small, upsert=True
                )
                
                # 2. Write large layout fields to layouts collection
                has_layout = any(k in data for k in _LAYOUT_FIELDS)
                if has_layout and self._layouts_collection is not None:
                    layout_doc = self._serialize_layout(doc_id, data)
                    self._layouts_collection.replace_one(
                        {"_id": doc_id}, layout_doc, upsert=True
                    )

                # Check and enforce 10-document FIFO limit for the user
                user_id = data.get("user_id")
                if user_id:
                    self._enforce_limit(user_id)
            except Exception as e:
                print(f"[MongoDB] Write error for {doc_id}: {e}")
        elif save_to_db and self._collection is None:
            # If MongoDB is local-only fallback, enforce cache limit
            user_id = data.get("user_id")
            if user_id:
                self._enforce_limit(user_id)

    def get(self, doc_id: str) -> dict | None:
        """Get document data (in-memory first, MongoDB fallback)."""
        if doc_id in self._cache:
            return self._cache[doc_id]
        if self._collection is not None:
            try:
                mongo_doc = self._collection.find_one({"_id": doc_id})
                if mongo_doc:
                    doc = self._deserialize_small(mongo_doc)
                    
                    if self._layouts_collection is not None:
                        layout_doc = self._layouts_collection.find_one({"_id": doc_id})
                        if layout_doc:
                            doc.update(self._deserialize_layout(layout_doc))
                    
                    self._cache[doc_id] = doc
                    return doc
            except Exception as e:
                print(f"[MongoDB] Read error for {doc_id}: {e}")
        return None

    def update(self, doc_id: str, updates: dict):
        """Update specific fields of a document using partial updates."""
        cached = self._cache.get(doc_id)
        if cached is None:
            cached = self.get(doc_id)
        if cached is None:
            return
        cached.update(updates)
        self._cache[doc_id] = cached

        if self._collection is None:
            return

        try:
            now = datetime.now(timezone.utc)
            
            small_updates = {}
            layout_updates = {}

            for k, v in updates.items():
                if k in _LAYOUT_FIELDS:
                    if v is not None:
                        if isinstance(v, (dict, list)):
                            layout_updates[k] = json.dumps(v, ensure_ascii=False)
                        else:
                            layout_updates[k] = v
                    else:
                        layout_updates[k] = None
                elif k in _JSON_FIELDS:
                    small_updates[k] = json.dumps(v, ensure_ascii=False) if v is not None else None
                elif isinstance(v, (str, int, float, bool, type(None), bytes)):
                    small_updates[k] = v
                else:
                    small_updates[k] = json.dumps(v, ensure_ascii=False)

            if small_updates:
                small_updates["updated_at"] = now
                self._collection.update_one(
                    {"_id": doc_id},
                    {"$set": small_updates},
                    upsert=True
                )

            if layout_updates and self._layouts_collection is not None:
                layout_updates["updated_at"] = now
                self._layouts_collection.update_one(
                    {"_id": doc_id},
                    {"$set": layout_updates},
                    upsert=True
                )

        except Exception as e:
            print(f"[MongoDB] Update error for {doc_id}: {e}")

    def register_user(self, username: str, password_hash: str) -> bool:
        """Register a new user. Returns True on success, False if user already exists."""
        if self._users_collection is not None:
            try:
                existing = self._users_collection.find_one({"_id": username})
                if existing:
                    return False
                user_doc = {
                    "_id": username,
                    "password_hash": password_hash,
                    "created_at": datetime.now(timezone.utc)
                }
                self._users_collection.insert_one(user_doc)
                self._users_cache[username] = user_doc
                return True
            except Exception as e:
                print(f"[MongoDB] Register error for {username}: {e}")

        if username in self._users_cache:
            return False
        user_doc = {
            "_id": username,
            "password_hash": password_hash,
            "created_at": datetime.now(timezone.utc)
        }
        self._users_cache[username] = user_doc
        return True

    def authenticate_user(self, username: str, password_hash: str) -> bool:
        """Verify user credentials. Returns True on success."""
        if self._users_collection is not None:
            try:
                user_doc = self._users_collection.find_one({"_id": username})
                if user_doc:
                    return user_doc.get("password_hash") == password_hash
                return False
            except Exception as e:
                print(f"[MongoDB] Auth error for {username}: {e}")

        user_doc = self._users_cache.get(username)
        if user_doc:
            return user_doc.get("password_hash") == password_hash
        return False

    def get_user_documents(self, user_id: str) -> list[dict]:
        """Get the list of documents for a user, sorted by updated_at descending."""
        if self._collection is not None:
            try:
                projection = {"pdf_bytes": 0, "middle_json": 0, "translated_middle": 0, "markdown": 0}
                cursor = self._collection.find({"user_id": user_id}, projection).sort("updated_at", -1)
                docs = []
                for mongo_doc in cursor:
                    doc = self._deserialize_from_mongo(mongo_doc)
                    docs.append(doc)
                return docs
            except Exception as e:
                print(f"[MongoDB] Error fetching documents for {user_id}: {e}")

        user_docs = [
            doc for doc in self._cache.values()
            if doc.get("user_id") == user_id and doc.get("save_to_db", True)
        ]
        user_docs.sort(key=lambda x: x.get("updated_at", datetime.now(timezone.utc)), reverse=True)
        return user_docs

    def _enforce_limit(self, user_id: str):
        """Enforce 10-document FIFO limit for a specific user."""
        # 1. Enforce in MongoDB
        if self._collection is not None:
            try:
                user_docs = list(self._collection.find({"user_id": user_id}, {"_id": 1}).sort("updated_at", 1))
                if len(user_docs) > 10:
                    num_to_delete = len(user_docs) - 10
                    for i in range(num_to_delete):
                        old_doc = user_docs[i]
                        old_id = old_doc["_id"]
                        self._collection.delete_one({"_id": old_id})
                        if old_id in self._cache:
                            del self._cache[old_id]
                        print(f"[MongoDB] FIFO Eviction: Deleted oldest document {old_id} for user {user_id}")
            except Exception as e:
                print(f"[MongoDB] Error enforcing limit for {user_id}: {e}")

        # 2. Enforce in local cache
        user_cache_docs = [
            (doc_id, doc) for doc_id, doc in self._cache.items()
            if doc.get("user_id") == user_id and doc.get("save_to_db", True)
        ]
        if len(user_cache_docs) > 10:
            user_cache_docs.sort(key=lambda x: x[1].get("updated_at", datetime.now(timezone.utc)))
            num_to_delete = len(user_cache_docs) - 10
            for i in range(num_to_delete):
                old_id = user_cache_docs[i][0]
                if old_id in self._cache:
                    del self._cache[old_id]
                print(f"[Cache] FIFO Eviction: Deleted oldest cached document {old_id} for user {user_id}")
