import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
import re
import subprocess
from datetime import datetime
from pydub import AudioSegment
from .audio_processor import AudioProcessor
from qwen_omni_utils import process_mm_info


class VideoProcessor:
    def __init__(self, audio_processor, MAX_NEW_TOKEN=2048):
        self.omni_processor = audio_processor
        self.RETURN_AUDIO = False
        self.MAX_NEW_TOKEN = MAX_NEW_TOKEN
        print("[SUCCESS] VideoProcessor inited.")

    def generate_caption(self, file_path, use_audio_in_video=True):
        messages = [{"role": "system",
                    "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
                    {"role": "user",
                     "content": [{"type": "video", "video": file_path, "max_pixels": 108000},
                                 {"type": "text", "text": "Describe this video."}]}]

        text = self.omni_processor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = self.omni_processor.processor(text=text, audio=audios, images=images, videos=videos,
                                               return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video)
        inputs = inputs.to(self.omni_processor.omni_model.device).to(self.omni_processor.omni_model.dtype)

        output = self.omni_processor.omni_model.generate(**inputs,
                                                         use_audio_in_video=use_audio_in_video,
                                                         return_audio=self.RETURN_AUDIO,
                                                         max_new_tokens=self.MAX_NEW_TOKEN)
        text = self.omni_processor.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
    
    def process_audio(self, file_path, max_seg_length=30):
        try:
            transcribe_results = self.omni_processor.asr_model.process_file(file_path)
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
                seg["audio_caption"] = self.omni_processor.generate_caption(temp_file)
            return segments
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}'s audio: {e}")
            return None
    
    def process_vision(self, file_path, segments):
        for seg in segments:
            temp_file = f"temp{os.path.splitext(file_path)[1]}"
            cmd = [
                "ffmpeg", "-y",
                "-i", file_path,
                "-ss", str(seg["start_time"]),
                "-t", str(seg["end_time"] - seg["start_time"]),
                temp_file
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            seg["vision_caption"] = self.generate_caption(temp_file, use_audio_in_video=False)
        return segments

    def process_file(self, file_path, max_seg_length=30):
        timestamp = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")

        try:
            caption = self.generate_caption(file_path, use_audio_in_video=True)
            content = [{
                "file_path": file_path,
                "file_type": "video",
                "content": caption,
                "timestamp": timestamp
            }]

            segments = self.process_audio(file_path, max_seg_length)
            segments = self.process_vision(file_path, segments)
            
            def seconds_to_mmss(seconds):
                mm, ss = divmod(int(seconds), 60)
                return f"{mm:02d}:{ss:02d}"

            text = ""
            for seg in segments:
                for item in seg["transcription"]:
                    text += f"[{seconds_to_mmss(item["start_time"])} - {seconds_to_mmss(item["end_time"])}] {item["text"]}\n"
                text += f"vision_summary: {seg["vision_caption"]}\n"
                text += f"audio_summary: {seg["audio_caption"]}"
                content.append({
                    "file_path": file_path,
                    "file_type": "video",
                    "content": text,
                    "timestamp": timestamp,
                    "is_segment": True,
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"]
                })
            return content

        except Exception as e:
            print(f"[ERROR] Failed to process video {file_path} {e}")
            return None


if __name__ == "__main__":
    audio_processor = AudioProcessor(dicow_port=8001, mllm_name="Qwen/Qwen2.5-Omni-7B", cache_dir="/data1/cxy/models")
    video_processor = VideoProcessor(audio_processor)
    result = video_processor.process_file("../data/XPIsnaE-vUM.mp4")
    print(result)
