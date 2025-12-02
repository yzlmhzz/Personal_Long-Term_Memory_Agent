#coding=gbk
import os
import re
import datetime
import streamlit as st
from memory_agent import MemoryAgent


data_dir="data"
db_dir="memory/chroma_db_storage"
dicow_port=8001
vl_model="Qwen/Qwen3-VL-8B-Instruct"
omni_model="Qwen/Qwen2.5-Omni-7B"
llm_model="Qwen/Qwen3-8B"
rerank_model="BAAI/bge-reranker-base"
cache_dir="/data1/cxy/models"


st.set_page_config(
    page_title="Personal Long-Term Memory Agent",
    layout="wide"
)
st.markdown(
    "<h1 style='text-align: center;'>Personal Long-Term Memory Agent</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray;'>support data type: images, videos, audios and docs (pdf/docx/txt)</p>", 
    unsafe_allow_html=True
)


@st.cache_resource
def get_agent():
    return MemoryAgent(data_dir=data_dir,
                       db_dir=db_dir,
                       dicow_port=dicow_port,
                       vl_model=vl_model,
                       omni_model=omni_model,
                       llm_model=llm_model,
                       rerank_model=rerank_model,
                       cache_dir=cache_dir)

try:
    agent = get_agent()
except Exception as e:
    st.error(f"Unable to open the agent: {e}")
    st.stop()


def update_logs(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    current_logs.append(log_line)
    log_string = "\n".join(current_logs)
    log_container.code(log_string, language="bash")

def save_uploaded_file(uploaded_file):
    save_dir = os.path.join(os.path.abspath(data_dir), datetime.datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def parse_response(text):
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        thought = match.group(1).strip()
        answer = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return thought, answer
    else:
        return None, text

def render_assistant_message(content, sources=None, contexts=None):
    thought, answer = parse_response(content)

    st.markdown(answer)

    has_sources = sources and len(sources) > 0
    if thought or has_sources:
        with st.expander("Thinking & Sources", expanded=False):
            if thought:
                st.markdown("#### Thoughts")
                st.info(thought)
    
            if thought and has_sources:
                st.divider()

            if has_sources:
                st.markdown("#### References")
                for _, (src, context) in enumerate(zip(sources, contexts)):
                    with st.container(border=True):
                        file_path = src.get("file_path", "UNKOWN")
                        file_type = src.get("file_type", "UNKOWN")

                        col_info, col_segment = st.columns([8, 2])
                        with col_info:
                            st.write(f"**{os.path.basename(file_path)}**")
                        
                        with col_segment:
                            if "start_time" in src and "end_time" in src:
                                start_mm, start_ss = divmod(int(src["start_time"]), 60)
                                end_mm, end_ss = divmod(int(src["end_time"]), 60)
                                st.caption(f"{start_mm:02d}:{start_ss:02d} - {end_mm:02d}:{end_ss:02d}")

                        full_path = os.path.join(data_dir, file_path)
                        if os.path.exists(full_path):
                            if "image" in file_type:
                                st.image(full_path, use_container_width=True)
                            elif ("video" in file_type or "audio" in file_type):
                                if "start_time" in src:
                                    st.video(full_path, start_time=int(src["start_time"]))
                                else:
                                    st.video(full_path)

                        st.caption(context[:100])


with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of retrievals (Top-K)", 1, 10, 4)
    
    st.header("System Logs")
    log_container = st.empty()
    current_logs = []
    
    st.divider()
    st.header("Add new data")
    with st.expander("upload", expanded=False):
        uploaded_file = st.file_uploader(
            "image/audio/video/document",
            type=['jpg', 'png', 'mp3', 'm4a', 'mp4', 'pdf', 'docx', 'txt']
        )
        
        if uploaded_file is not None:
            if st.button("Enter"):
                file_path = save_uploaded_file(uploaded_file)
                try:
                    with st.spinner("Processing ..."):
                        if agent.add(file_path):
                            agent = get_agent()
                            update_logs(f"[INFO] {file_path} upload success")
                        else:
                            update_logs(f"[ERROR] Failed to upload {file_path}")
                except Exception as e:
                    st.error(f"Failed to process: {e}")
                    update_logs(f"[ERROR] Failed to upload {e}")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your memory agent. You can ask me: \"When did I visit Shanghai?\" or \"Look for the meeting minutes about the budget.\""}
    ]
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            sources = msg.get("sources", [])
            contexts = msg.get("contexts", [])
            render_assistant_message(msg["content"], sources, contexts)
        else:
            st.markdown(msg["content"])


if prompt := st.chat_input("Input your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = ""
        
        with st.spinner("running ..."):
            try:
                result = agent.ask(prompt, top_k=top_k, verbose_func=update_logs)

                render_assistant_message(
                    result["answer"], 
                    result.get("sources"), 
                    result.get("contexts")
                )

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "contexts": result.get("contexts", [])
                })

            except Exception as e:
                update_logs(f"[ERROR] {e}")
                st.error(f"[ERROR] {e}")
