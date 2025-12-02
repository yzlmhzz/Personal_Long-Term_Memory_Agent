import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import io
import fitz
import shutil
import docx2txt
from PIL import Image
from datetime import datetime
from .image_processor import ImageProcessor
from langchain_community.document_loaders import TextLoader


class DocumentProcessor:
    def __init__(self, image_processor):
        self.image_processor = image_processor
        print("[SUCCESS] DocumentProcessor inited.")

    def process_pdf(self, file_path):
        timestamp = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
        doc = fitz.open(file_path)

        content = []
        for page_num, page in enumerate(doc):
            text = page.get_text()

            image_list = page.get_images(full=True)
            image_captions = ""
            if image_list:
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.width < 100 or image.height < 100:
                        continue

                    temp_path = "temp.png"
                    image.save(temp_path)
                    caption = self.image_processor.generate_caption(temp_path)
                    if caption:
                        image_captions += f"\nIMAGE {img_index + 1}: {caption}\n"

            page_content = text + image_captions
            content.append({
                "file_path": file_path,
                "file_type": "document",
                "content": page_content,
                "timestamp": timestamp,
                "is_segment": True,
                "page_number": page_num + 1,
            })

        return content

    def process_docx(self, file_path):
        timestamp = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
        temp_img_dir = "temp_docx_images"
        if os.path.exists(temp_img_dir):
            shutil.rmtree(temp_img_dir)
        os.makedirs(temp_img_dir)

        try:
            text = docx2txt.process(file_path, temp_img_dir)

            image_captions = ""
            if os.path.exists(temp_img_dir):
                img_files = os.listdir(temp_img_dir)
                
                if img_files:
                    for img_index, img_file in enumerate(img_files):
                        img_full_path = os.path.join(temp_img_dir, img_file)
                        caption = self.image_processor.generate_caption(img_full_path)
                        if caption:
                            image_captions += f"\nIMAGE {img_index + 1}: {caption}\n"

            content = text + image_captions
            return [{
                "file_path": file_path,
                "file_type": "document",
                "content": content,
                "timestamp": timestamp
            }]

        except Exception as e:
            print(f"[ERROR] Failed to process document {file_path}: {e}")
            return []

        finally:
            if os.path.exists(temp_img_dir):
                shutil.rmtree(temp_img_dir)
        
    def process_txt(self, file_path):
        timestamp = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
        loader = TextLoader(file_path, encoding="utf-8")
        
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])

        return [{
            "file_path": file_path,
            "file_type": "document",
            "content": content,
            "timestamp": timestamp
        }]

    def process_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                content = self.process_pdf(file_path)
            elif ext == ".docx":
                content = self.process_docx(file_path)
            elif ext == ".txt":
                content = self.process_txt(file_path)
            else:
                print(f"[ERROR] Unsupported document format: {ext}")
                return None
            return content
        except Exception as e:
            print(f"[ERROR] Failed to process document {file_path}: {e}")
            return None


if __name__ == "__main__":
    image_processor = ImageProcessor(model_name="Qwen/Qwen3-VL-8B-Instruct", cache_dir="/data1/cxy/models")
    document_processor = DocumentProcessor(image_processor)
    result = document_processor.process_file("../data/study.docx")
    print(result)
