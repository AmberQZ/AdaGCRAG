import pandas as pd
from torch.utils.data import Dataset
import os
import asyncio
from Core.Utils.Evaluation import Evaluator
from tqdm import tqdm    
import sys
import asyncio
from tqdm import trange
from minirag import MiniRAG, QueryParam
from minirag.llm import (
    gpt_4o_mini_complete,
    hf_embed,
    ollama_model_complete,
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RAGQueryDataset(Dataset):
    def __init__(self,data_dir):
        super().__init__()
      
        self.qa_path = os.path.join(data_dir, "Question.json")        
        self.dataset = pd.read_json(self.qa_path, lines=True, orient="records")
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question = self.dataset.iloc[idx]["question"]
        answer = self.dataset.iloc[idx]["answer"]
        other_attrs = self.dataset.iloc[idx].drop(["answer", "question"])
        return {"id": idx, "question": question, "answer": answer, **other_attrs}


async def async_embedding_func(texts: list[str], embedding_model, embedding_tokenizer):
    return await hf_embed(texts, tokenizer=embedding_tokenizer, embed_model=embedding_model)

embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL)
embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


def extract_after_think(text):
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()  # fallback: return original if tag not found

def wrapper_query(query_dataset, result_dir, dataset_name, llm_model):
    all_res = []

    dataset_len = len(query_dataset)

    for i in tqdm(range(dataset_len), desc="Processing queries"):
        query = query_dataset[i]
        res = rag.query(query['question'], param=QueryParam(mode="mini"))
        res=extract_after_think(res)
        if res.startswith("Answer:"):
            answer_text = res[len("Answer:"):].strip()
        else:
            answer_text = res.strip()

        query["output"] = answer_text
        all_res.append(query)

    all_res_df = pd.DataFrame(all_res)
    filename = f"{dataset_name}_{llm_model}_results.json"
    print(filename)
    save_path = os.path.join(result_dir, filename)
    all_res_df.to_json(save_path, orient="records", lines=True)
    return save_path


async def wrapper_evaluation(path, dataset_name, result_dir, llm_model):
    eval = Evaluator(path, dataset_name)
    res_dict = await eval.evaluate()
    filename = f"{dataset_name}_{llm_model}_metrics.json"
    save_path = os.path.join(result_dir, filename)
    with open(save_path, "w") as f:
        f.write(str(res_dict))


if __name__ == "__main__":
    dataset_name = "multihop-rag"
    query_dataset = RAGQueryDataset(data_dir=f"/mnt/g/PminiRAG_test/datasets/{dataset_name}")    
    WORKING_DIR = f"./processed_data/qwen3_4b_{dataset_name}"
    LLM_MODEL = "qwen3:4b"

    print("USING LLM:", LLM_MODEL)
    rag = MiniRAG(
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

    save_path = wrapper_query(query_dataset, "res", dataset_name, LLM_MODEL.replace(":", "_"))
    asyncio.run(wrapper_evaluation(save_path, dataset_name, "res", LLM_MODEL.replace(":", "_")))