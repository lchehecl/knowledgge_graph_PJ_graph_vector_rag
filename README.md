# 🧠 Medical Imaging Knowledge Graph + RAG Pipeline

This repository implements a **knowledge graph (KG)-enhanced Retrieval-Augmented Generation (RAG)** system for answering questions about medical imaging research (e.g., MRI-based brain tumor segmentation), powered by:
- **Neo4j** for graph storage,
- **BGE-M3** embeddings for semantic retrieval,
- **Qwen-Turbo** (via DashScope API) for generation.

Designed for researchers in medical AI, especially those working with papers from *IEEE TMI*, *Medical Image Analysis*, etc.

---

## 📁 Project Structure

> ⚠️ `graph_data.jsonl` (raw export, no embeddings) is an intermediate file, not committed.

---

## 🛠️ Setup & Usage

### Prerequisites
- Python ≥ 3.9
- Neo4j Aura (or local Neo4j instance)
- DashScope API Key ([apply here](https://dashscope.console.aliyun.com/))
- GPU recommended (for embedding inference)

### 1. Install Dependencies
```bash
pip install neo4j sentence-transformers scikit-learn pandas torch dashscope tqdm
```

### 2. Configure Neo4j & API Keys
# export_neo4j.py
URI = "your_neo4j_uri"
AUTH = ("neo4j", "your_password")

# run_qwen_rag.py
DASHSCOPE_API_KEY = "sk-xxxxxx"  # ← Replace!

### 3. Run the Pipeline
🔹 Export KG
python export_neo4j.py → graph_data.jsonl

🔹 Embed with BGE-M3
python re_embed_with_bge_m3.py → graph_data_bge-m3.jsonl

🔹 Run RAG Demo
python run_qwen_rag.py
Compare vanilla vs RAG answers
