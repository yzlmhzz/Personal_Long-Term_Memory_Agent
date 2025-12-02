#coding=gbk
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MODELSCOPE_CACHE"] = "/data1/cxy/models"
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_path) 
if project_root not in sys.path:
    sys.path.append(project_root)
import re
import json
import torch
from datetime import datetime
from memory.memory import Memory
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification


class MemoryAgent:
    def __init__(self,
                 data_dir="data",
                 db_dir="memory/chroma_db_storage",
                 dicow_port=8001,
                 vl_model="Qwen/Qwen3-VL-8B-Instruct",
                 omni_model="Qwen/Qwen2.5-Omni-7B",
                 llm_model="Qwen/Qwen3-8B",
                 rerank_model="BAAI/bge-reranker-base",  # "BAAI/bge-reranker-v2-m3"
                 cache_dir="/data1/cxy/models",
                 max_new_token=2048):
        self.data_dir = data_dir
        self.memory = Memory(data_dir=data_dir,
                             db_dir=db_dir,
                             dicow_port=dicow_port,
                             vl_model=vl_model,
                             omni_model=omni_model,
                             cache_dir=cache_dir)
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(rerank_model, cache_dir=cache_dir)
        self.reranker = AutoModelForSequenceClassification.from_pretrained(rerank_model, cache_dir=cache_dir)
        self.reranker.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model, cache_dir=cache_dir)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            cache_dir=cache_dir,
            dtype="auto",
            device_map="auto"
        )
        self.history = []
        self.MAX_NEW_TOKEN=max_new_token
        print("[INFO] Memory agent inited.")
    
    def parse_query(self, query):
        current_date = datetime.now().strftime("%Y-%m-%d")

        prompt = f"""
        Current Date: {current_date}

        You are an intelligent Query Parser for a Personal Memory Agent. 
        Your task is to extract search parameters from the user's natural language query to filter a vector database (ChromaDB).
        
        ### Date Logic (CRITICAL):
        1. Filter (date_range): Use this ONLY if the user is looking for files at a specific time.
        2. Content (keywords): If the user asks about an event, plan, or schedule associated with a date, treat the date as a KEYWORD to search within the text. Do NOT set date_range in this case (because the document describing the future plan was created in the past).

        Please analyze the user's query and extract the following information in JSON format:

        1. "keywords": A list of the most important semantic keywords (entities, actions, objects) for vector retrieval.
        2. "file_type": A list of file types implied by the user. 
        - Allowed values: ["image", "video", "audio", "document"].
        - If the user asks for "photos" or "pictures", map to "image".
        - If the user asks for "meeting notes" or "papers", map to "document".
        - If the user mentions "songs", map to "audio".
        - If not specified, return an empty list [].
        3. "date_range": The time range implied by the query.
        - Format: "YYYY-MM-DD"
        - "start": Start date (inclusive).
        - "end": End date (inclusive).
        - If no time is specified, return null for both.
        
        ### Example:
        User Query: "Summarize the meeting minutes about the budget."
        JSON Output:
        {{
            "keywords": ["meeting minutes", "budget", "finance"],
            "file_type": ["document"],
            "date_range": {{
                "start": null,
                "end": null
            }}
        }}
        
        ### Your Turn:
        User Query: "{query}"
        JSON Output:
        """

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        generated_ids = self.llm_model.generate(**inputs, max_new_tokens=self.MAX_NEW_TOKEN)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist() 

        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        try:
            pattern = r"<think>(.*?)</think>"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                response = re.sub(pattern, "", response, flags=re.DOTALL).strip()
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    filters = json.loads(json_str)
                    return filters
                else:
                    print("[WARN] No JSON found in parser response")
                    return None
        except Exception as e:
            print(f"[ERROR] JSON parsing failed: {e}")
            return None

    def retrieve_context(self, filters=None, top_k=5):
        if self.log:
            self.log("[INFO] Searching ...")
        query = " ".join(filters["keywords"])
        print(filters)
        results = self.memory.search([query], top_k=top_k * 5, filters=filters)

        raw_docs = []
        raw_metas = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]

                if dist > 1.5: 
                    continue
                raw_docs.append(doc)
                raw_metas.append(meta)
        
        if not raw_docs:
            return [], []

        if self.log:
            self.log("[INFO] Reranking ...")
        contexts = []
        sources = []

        pairs = [[query, doc] for doc in raw_docs]
        with torch.no_grad():
            inputs = self.reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            scores = self.reranker(**inputs, return_dict=True).logits.view(-1, ).float()
        results = list(zip(raw_docs, raw_metas, scores))
        results.sort(key=lambda x: x[2], reverse=True)
        for doc, meta, score in results[:top_k]:
            # print(f"[Score: {score:.4f}] {doc[:50]}...")
            if score > -2.0:
                contexts.append(doc)
                sources.append(meta)
        
        if not contexts:
            contexts.append(results[0][0])
            sources.append(results[0][1])
        return contexts, sources

    def generate_answer(self, query, contexts, sources):
        context_str = ""
        if contexts:
            for context, source in zip(contexts, sources):
                context_str += f"Source: {source["file_path"]}:\n{context}\n-----\n"
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""
        Current Date: {current_date}

        You are a helpful assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context_str}
        
        Question: {query}
        """

        self.history.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        generated_ids = self.llm_model.generate(**inputs, max_new_tokens=self.MAX_NEW_TOKEN)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist() 

        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        self.history.append({"role": "assistant", "content": response})
        return response

    def ask(self, query, top_k=2, verbose_func=None):
        self.log = verbose_func

        filters = self.parse_query(query)
        if not filters:
            filters = {"keywords": [query]}
        # else:
        #     filters["keywords"].append(query)
        contexts, sources = self.retrieve_context(filters, top_k=top_k)
        if self.log:
            self.log("[INFO] Retrieval completed")
        if self.log:
            self.log("[INFO] Inferencing ...")
        answer = self.generate_answer(query, contexts, sources)
        if self.log:
            self.log("[INFO] Inference completed")

        return {
            "answer": answer,
            "sources": sources,
            "contexts": contexts
        }
    
    def add(self, file_path):
        return self.memory.add(file_path)


if __name__ == "__main__":
    agent = MemoryAgent()
    res = agent.ask("帮我找提到模态对齐的文件")
    print(res["answer"])
    print(res["sources"])
