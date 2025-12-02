import os
# os.environ["HF_HOME"] = "data1/caixinyue/models"
# os.environ["CHROMA_CACHE_DIR"] = "data1/caixinyue/models"
import sys
import json
# current_path = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_path) 
# if project_root not in sys.path:
#     sys.path.append(project_root)
from tqdm import tqdm

from .vector_db import LocalVectorDB
from processors.image_processor import ImageProcessor
from processors.audio_processor import AudioProcessor
from processors.video_processor import VideoProcessor
from processors.document_processor import DocumentProcessor


def get_all_files_on_disk(data_dir):
    disk_files = set()
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.startswith("."):
                continue
            
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, start=data_dir)
            rel_path = rel_path.replace("\\", "/")

            disk_files.add(rel_path)
    return disk_files


class Memory:
    def __init__(self, data_dir, db_dir, dicow_port, vl_model, omni_model, cache_dir):
        self.data_dir = os.path.abspath(data_dir)

        self.image_processor = ImageProcessor(vl_model=vl_model, cache_dir=cache_dir)
        self.audio_processor = AudioProcessor(dicow_port=dicow_port, omni_model=omni_model, cache_dir=cache_dir)
        self.video_processor = VideoProcessor(self.audio_processor)
        self.document_processor = DocumentProcessor(self.image_processor)

        self.supported_extensions = {
            "image": [".jpg", ".jpeg", ".png", ".bmp"],
            "audio": [".mp3", ".wav", ".m4a"],
            "document": [".pdf", ".docx", ".txt"],
            "video": [".mp4"]
        }

        os.makedirs(self.data_dir, exist_ok=True)
        print("[INFO] Scanning the data pool ...")
        files_on_disk = get_all_files_on_disk(self.data_dir)

        self.db = LocalVectorDB(persist_dir=db_dir)
        print("[INFO] Reading the database ...")
        files_in_db = self.db.get_all_file_paths()

        files_to_add = list(files_on_disk - files_in_db)
        files_to_delete = list(files_in_db - files_on_disk)

        if files_to_delete:
            self.db.delete_by_paths(files_to_delete)

        add_cnt = 0
        for rel_path in tqdm(files_to_add, desc="Processing New Files"):
            try:
                abs_file_path = os.path.join(data_dir, rel_path)
                if self.add(abs_file_path):
                    add_cnt += 1
            except Exception as e:
                print(f"[ERROR] Failed to process file {abs_file_path}: {e}")

        print("[INFO] Done simultaneously")
        print(f"[INFO] Addition: {add_cnt} files")
        print(f"[INFO] Delete: {len(files_to_delete)} files")

    def print(self):
        self.db.print()
    
    def search(self, query, top_k=5, filters=None):
        return self.db.search(query, top_k, filters)
    
    def add(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()

        result = []
        if ext in self.supported_extensions["image"]:
            result = self.image_processor.process_file(file_path)
        elif ext in self.supported_extensions["audio"]:
            result = self.audio_processor.process_file(file_path)
        elif ext in self.supported_extensions["video"]:
            result = self.video_processor.process_file(file_path)
        elif ext in self.supported_extensions["document"]:
            result = self.document_processor.process_file(file_path)

        if result:
            for item in result:
                item["file_path"] = os.path.relpath(file_path, start=self.data_dir).replace("\\", "/")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            self.db.add_data(result)
            return True
        return False


if __name__ == "__main__":
    memory = Memory(data_dir="../data",
                    db_dir="chroma_db_storage",
                    dicow_port=8001,
                    vl_model="Qwen/Qwen3-VL-8B-Instruct",
                    omni_model="Qwen/Qwen2.5-Omni-7B",
                    cache_dir="/data1/cxy/models")
    memory.print()
