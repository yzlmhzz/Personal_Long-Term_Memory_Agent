import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["MODELSCOPE_CACHE"] = "/data1/cxy/models"
import re
import requests
import subprocess
from datetime import datetime
from pydub import AudioSegment
from faster_whisper import WhisperModel
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


class DicowProcessor:
    def __init__(self, api_url="http://localhost:8001/transcribe"):
        self.api_url = api_url

    def process_file(self, file_path):
        abs_path = os.path.abspath(file_path)
        try:
            payload = {"file_path": abs_path}
            response = requests.post(self.api_url, json=payload, timeout=600)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    data = result.get("data", [])
                    return data
                else:
                    print("[ERROR] dicow failed to tanscribe")
                    return []
            else:
                print(f"[ERROR] dicow failed to tanscribe {response.status_code} {response.text}")
                return []
                
        except Exception as e:
            print(f"[ERROR] failed to connect to dicow {e}")
            return []


class AudioProcessor:
    def __init__(self, dicow_port, omni_model, cache_dir, max_new_token=2048):
        self.asr_model = DicowProcessor(api_url=f"http://localhost:{dicow_port}/transcribe")
        # self.asr_model = WhisperModel("large-v3", device="cuda", compute_type="float16", download_root=cache_dir)

        self.processor = Qwen2_5OmniProcessor.from_pretrained(omni_model, cache_dir=cache_dir)
        self.omni_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            omni_model,
            cache_dir=cache_dir,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.USE_AUDIO_IN_VIDEO = False
        self.omni_model.disable_talker()
        self.RETURN_AUDIO = False
        self.MAX_NEW_TOKEN=max_new_token
        print("[SUCCESS] AudioProcessor inited.")

    def generate_caption(self, audio_path=None):
        messages = [{"role": "system",
                    "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
                    {"role": "user",
                     "content": [{"type": "audio", "audio": audio_path},
                                 # {"type": "text", "text": "Describe this audio, summarizing both the background sounds and the main events."}]}]
                                 {"type": "text", "text": "Describe this audio, summarizing both the background sounds and the main events."}]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=self.USE_AUDIO_IN_VIDEO)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True, use_audio_in_video=self.USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(self.omni_model.device).to(self.omni_model.dtype)

        output = self.omni_model.generate(**inputs,
                                    use_audio_in_video=self.USE_AUDIO_IN_VIDEO,
                                    return_audio=self.RETURN_AUDIO,
                                    max_new_tokens=self.MAX_NEW_TOKEN)
        text = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        matches = re.findall(r"assistant\n(.*?)(?=\nuser|$)", text[0], flags=re.DOTALL)
        if matches:
            response = matches[-1].strip()
        else:
            response = text[0]
        return response

    def build_segments(self, asr_segments, duration, max_seg_length):
        segments = []
        last_asr_end = 0
        for seg in asr_segments:
            start = seg["start_time"]
            end = seg["end_time"]

            # non-speech part
            if start > last_asr_end:
                if segments and start - segments[-1]["start_time"] < max_seg_length:
                    segments[-1]["end_time"] = seg["start_time"]
                else:
                    segments.append({"start_time": last_asr_end,
                                     "end_time": seg["start_time"],
                                     "transcription": []})

            # speech part
            if segments and start < (segments[-1]["end_time"]):
                segments[-1]["end_time"] = seg["end_time"]
                segments[-1]["transcription"].append({"start_time": seg["start_time"],
                                                      "end_time": seg["end_time"],
                                                      "text": seg["text"]})
            elif segments and end - (segments[-1]["start_time"]) < max_seg_length:
                segments[-1]["end_time"] = seg["end_time"]
                segments[-1]["transcription"].append({"start_time": seg["start_time"],
                                                      "end_time": seg["end_time"],
                                                      "text": seg["text"]})
            else:
                segments.append({"start_time": seg["start_time"],
                                 "end_time": seg["end_time"],
                                 "transcription": [{"start_time": seg["start_time"],
                                                    "end_time": seg["end_time"],
                                                    "text": seg["text"]}]})
            last_asr_end = end

        # last non-speech part
        if duration > last_asr_end:
            if segments and duration - (segments[-1]["start_time"]) < max_seg_length:
                segments[-1]["end_time"] = duration
            else:
                segments.append({"start_time": last_asr_end,
                                 "end_time": duration,
                                 "transcription": []})

        for seg_num in range(1, len(segments)):
            assert (segments[seg_num]["start_time"]) == (segments[seg_num - 1]["end_time"])
        return segments
    
    def process_file(self, file_path, max_seg_length=30):
        timestamp = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
    
        try:
            caption = self.generate_caption(file_path)
            content = [{
                "file_path": file_path,
                "file_type": "audio",
                "content": caption,
                "timestamp": timestamp
            }]
            
            transcribe_results = self.asr_model.process_file(file_path)
            duration = round(len(AudioSegment.from_file(file_path)) / 1000)
            segments = self.build_segments(transcribe_results, duration, max_seg_length)
            
            for seg in segments:
                temp_file = f"temp{os.path.splitext(file_path)[1]}"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", file_path,
                    "-ss", str(seg["start_time"]),
                    "-t", str(seg["end_time"] - seg["start_time"]),
                    "-c", "copy",
                    temp_file
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                seg_caption = self.generate_caption(temp_file)
                
                def seconds_to_mmss(seconds):
                    mm, ss = divmod(int(seconds), 60)
                    return f"{mm:02d}:{ss:02d}"

                text = ""
                for item in seg["transcription"]:
                    text += f"[{seconds_to_mmss(item["start_time"])} - {seconds_to_mmss(item["end_time"])}] {item["text"]}\n"
                text += f"summary: {seg_caption}"
                content.append({
                    "file_path": file_path,
                    "file_type": "audio",
                    "content": text,
                    "timestamp": timestamp,
                    "is_segment": True,
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"]
                })
            return content
            # segments, info = self.asr_model.transcribe(
            #     file_path, 
            #     beam_size=5,
            #     vad_filter=True,
            #     vad_parameters=dict(min_silence_duration_ms=1000)
            # )
            # seg_list = list(segments) 
            # print(seg_list)

        except Exception as e:
            print(f"[ERROR] Failed to process audio {file_path}: {e}")
            return None


if __name__ == "__main__":
    # client = DicowProcessor(api_url="http://localhost:8001/transcribe")
    # result = client.process_file("/home/cxy/Personal_Long-Term_Memory_Agent/data/KHCfnYQSEmg.m4a")
    # print(result)

    audio_processor = AudioProcessor(dicow_port=8001, omni_model="Qwen/Qwen2.5-Omni-7B", cache_dir="/data1/cxy/models")
    result = audio_processor.process_file("../data/KHCfnYQSEmg.m4a")
    print(result)