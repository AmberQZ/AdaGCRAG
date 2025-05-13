# AdaGCRAG

AdaGCRAG is a lightweight and modular RAG framework that integrates chunk-based semantic retrieval with graph-based reasoning in a query-adaptive manner.

---

## 📁 Datasets

The `datasets/` directory includes the following:

- **HotpotQA** (`hotpotqa/`)| **Musique** (`musique/`)| **Mix** (`mix/`)| **Multihop-RAG** (`multihop-rag/`)| **Multihop-RAG-Summary** (`multihop-rag-summary/`)

---

## ⚙️ Setup and Installation

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure all datasets are placed correctly under the `datasets/` directory.

---

## 🚀 Usage

### Step 1: Build the Knowledge Graph

Construct the knowledge graph, extract entities, relations, and chunks, and embed them into the vector database:

```bash
python step1_offline_index.py
```

---

### Step 2: Test Specific QA Dataset

Run evaluation on Specific QA datasets:

```bash
python step2_sqa_evaluation.py
```

---

### Step 3: Test Abstract QA Dataset

Navigate to the Abstract QA evaluation directory and run the evaluation script:

```bash
cd step3_aq_evaluation/
python  python summary_eval.py --input_file1 file1.json --input_file2 file2.json --output_file_name output.csv
```

---

### Step 4: Single Question Answering

Test the system on a single question using:

```bash
python step4_singleQA.py
```

---

---

## ✨ Features

- **Knowledge Graph Construction**: Extracts and embeds entities, relations, and chunks.
- **Vector Database Integration**: Stores embeddings for efficient retrieval.
- **Specific QA Evaluation**: Evaluates performance on structured, task-specific questions.
- **Abstract QA Evaluation**: Supports evaluation on more open-ended, general QA tasks.
- **Single QA Testing**: Lightweight testing interface for debugging and experimentation.

---
