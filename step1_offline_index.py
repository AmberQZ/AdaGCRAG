import sys
import os
import pandas as pd
from tqdm import tqdm
import logging

logging.getLogger().setLevel(logging.WARNING)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from adarag import adarag
from adarag.llm import (
    gpt_4o_mini_complete,
    hf_embed,
    ollama_model_complete,
)

from adarag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="adarag")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./qwen3_4b_multihop-rag")
    parser.add_argument("--datapath", type=str, default="./datasets/multihop-rag/Corpus.json")
    args = parser.parse_args()
    return args


args = get_args()


if args.model == "llama3":
    LLM_MODEL = "llama3.1:8b"
elif args.model == "qwem":
    LLM_MODEL = "qwen3:4b"
else:
    print("Invalid model name")
    exit(1)

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
OUTPUT_PATH = args.outputpath
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = adarag(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
)

# Now indexing
corpus = pd.read_json(DATA_PATH, lines=True)
# for _, row in tqdm(corpus.iterrows(), total=len(corpus), desc="Inserting into RAG"):
#     rag.insert(row["context"])
full_context = "\n\n".join(corpus["context"].tolist())
# print(full_context)
rag.insert(full_context)
