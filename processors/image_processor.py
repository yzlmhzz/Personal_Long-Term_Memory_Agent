import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from datetime import datetime
from PIL import Image, ExifTags
from transformers import AutoModelForImageTextToText, AutoProcessor


class ImageProcessor:
    def __init__(self, vl_model, cache_dir, max_new_token=2048):
        self.processor = AutoProcessor.from_pretrained(vl_model, cache_dir=cache_dir)
        self.vl_model = AutoModelForImageTextToText.from_pretrained(
            vl_model,
            cache_dir=cache_dir,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.MAX_NEW_TOKEN=max_new_token
        print(f"[SUCCESS] ImageProcessor Inited.")

    def generate_caption(self, image_path):
        messages = [{"role": "user",
                    "content": [{"type": "image", "image": image_path},
                                {"type": "text", "text": "Describe this image."}]}]
        inputs = self.processor.apply_chat_template(messages,
                                                    tokenize=True,
                                                    add_generation_prompt=True,
                                                    return_dict=True,
                                                    return_tensors="pt")
        inputs = inputs.to(self.vl_model.device)

        generated_ids = self.vl_model.generate(**inputs, max_new_tokens=self.MAX_NEW_TOKEN)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]
        return response
    
    def get_timestamp(self, image_path):
        timestamp = datetime.fromtimestamp(os.path.getmtime(image_path)).strftime("%Y-%m-%d %H:%M:%S")
        try:
            info = Image.open(image_path)._getexif()
            if info:
                for tag, value in info.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    if decoded == "DateTimeOriginal":
                        timestamp = value
        except:
            pass
        return timestamp

    def process_file(self, file_path):
        try:
            timestamp = self.get_timestamp(file_path)
            caption = self.generate_caption(file_path)
            return [{
                "file_path": file_path,
                "file_type": "image",
                "content": caption,
                "timestamp": timestamp,
            }]
        except Exception as e:
            print(f"[ERROR] Failed to process image {file_path}: {e}")
            return None


if __name__ == "__main__":
    image_processor = ImageProcessor(vl_model="Qwen/Qwen3-VL-8B-Instruct", cache_dir="/data1/cxy/models")
    result = image_processor.process_file("../data/f7e3fddb06e5cbd3ac84d9bb5c7c9ed2.jpg")
    print(result)
