import pandas as pd
from torch.utils.data import Dataset
import os
import sys
import re
from adarag import adarag, QueryParam
from adarag.llm import (
    gpt_4o_mini_complete,
    hf_embed,
    ollama_model_complete,
)
from adarag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

async def async_embedding_func(texts: list[str], embedding_model, embedding_tokenizer):
    return await hf_embed(texts, tokenizer=embedding_tokenizer, embed_model=embedding_model)

embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL)
embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

# 1.zero-shot查询
import requests
def extract_after_think(text):
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()  # fallback: return original if tag not found
def query_ollama_model(LLM_MODEL,query: str):
    url = "http://localhost:11434/api/generate"  # Ollama 模型的默认地址
    prompt = f"{query}"
    
    payload = {
        "model": LLM_MODEL,  # 或使用其他你想要的模型
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()  # 错误处理
    return response.json()["response"].strip()  # 返回模型的答案

if __name__ == "__main__":

    WORKING_DIR = f"./processed_data/qwen3_4b_multihop-rag"
    LLM_MODEL = "qwen3:4b"
    print("USING LLM:", LLM_MODEL)

    rag = adarag(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,
    
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: async_embedding_func(texts, embedding_model, embedding_tokenizer),
      ),
  )
    query ="What coaching strategies are highlighted in the dataset, and how do they impact team performance?"
    res1 =query_ollama_model(LLM_MODEL,query)
    print("Zero-shot response:", extract_after_think(res1))

    res2 = rag.query(query=query, param=QueryParam(mode="mini"))
    print("MixGCRAG response:", res2)
    
