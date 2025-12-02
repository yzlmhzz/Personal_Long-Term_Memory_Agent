# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import uuid
import chromadb


class LocalVectorDB:
    def __init__(self, persist_dir="./chroma_db_storage"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="personal_memory",
            metadata={"hnsw:space": "cosine"}
        )
        print("[INFO] Database connection successful")
    
    def reset_db(self):
        self.client.delete_collection("personal_memory")
        print("[INFO] The database has been emptied")
    
    def print(self):
        results = self.collection.peek(limit=5)
        for i in range(len(results["ids"])):
            print(f"\n[ID]: {results["ids"][i]}")
            print(f"[Meta]: {results["metadatas"][i]}")
            print(f"[Text]: {results["documents"][i]}")

    def get_all_file_paths(self):
        existing_data = self.collection.get(include=["metadatas"])

        paths = set()
        for meta in existing_data.get("metadatas", []):
            if meta and "file_path" in meta:
                paths.add(meta["file_path"])
        return paths

    def delete_by_paths(self, file_paths_list):
        if not file_paths_list:
            return

        try:
            self.collection.delete(
                where={"file_path": {"$in": file_paths_list}}
            )
        except Exception as e:
            print(f"[ERROR] Failed to delete data: {e}")

    def _sanitize_metadata(self, raw_meta):
        clean_meta = {}
        for k, v in raw_meta.items():
            if v is None:
                continue

            if isinstance(v, list):
                clean_meta[k] = ",".join([str(i) for i in v])
            elif isinstance(v, bool):
                clean_meta[k] = str(v)
            elif isinstance(v, (str, int, float)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        return clean_meta

    def add_data(self, processed_data_list):
        if not processed_data_list:
            return

        documents = []
        metadatas = []
        ids = []
        for item in processed_data_list:
            content = item.get("content", "").strip()
            if not content:
                continue

            meta = item.copy()
            if "content" in meta:
                del meta["content"]
            meta = self._sanitize_metadata(meta)
    
            doc_id = str(uuid.uuid4())

            documents.append(content)
            metadatas.append(meta)
            ids.append(doc_id)

        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"[ERROR] Failed to write to the database : {e}")

    def search(self, query, top_k=5, filters=None):
        conditions = []
        if filters:
            if filters.get("file_type"):
                if len(filters["file_type"]) == 1:
                    conditions.append({"file_type": filters["file_type"][0]})
                else:
                    conditions.append({
                        "$or": [{"file_type": t} for t in filters["file_type"]]
                    })

            date_range = filters.get("date_range")
            if date_range:
                start_str = date_range.get("start")
                end_str = date_range.get("end")

                if start_str:
                    start_str = f"{start_str} 00:00:00"
                    conditions.append({"timestamp": {"$gte": start_str}})
                if end_str:
                    end_str = f"{end_str} 23:59:59"
                    conditions.append({"timestamp": {"$lte": end_str}})

        where_clause = None
        if len(conditions) > 1:
            where_clause = {"$and": conditions}
        elif len(conditions) == 1:
            where_clause = conditions[0]
        results = self.collection.query(
            query_texts=query,
            n_results=top_k,
            where=where_clause if where_clause else None
        )
        return results
