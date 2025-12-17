# graph_rag.py —— 使用 sentence-transformers + prompt 模式
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import torch

class TmiGraphRAG:
    def __init__(self, jsonl_path: str):
        self.nodes = []
        self.relationships = []
        self.node_id_to_index = {}
        self.node_embeddings = None
        self._load_graph(jsonl_path)
        self._build_embeddings()
        self._init_embedding_model()  # ← 延后加载模型

    def _load_graph(self, path: str):
        print("📥 正在加载 graph_data.jsonl ...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if obj["type"] == "node":
                        self.nodes.append(obj)
                        self.node_id_to_index[obj["id"]] = len(self.nodes) - 1
                    elif obj["type"] == "relationship":
                        self.relationships.append(obj)
                except Exception as e:
                    print(f"⚠️ 跳过无效行: {e}")
        print(f"✅ 加载完成: {len(self.nodes)} 个节点, {len(self.relationships)} 条关系")

    def _parse_embedding(self, emb):
        if emb is None:
            return None
        if isinstance(emb, str):
            try:
                emb = json.loads(emb.replace("'", '"').replace("array(", "[").replace(")", "]"))
            except:
                return None
        if isinstance(emb, list):
            return np.array(emb, dtype=np.float32)
        return None

    def _build_embeddings(self):
        print("🧮 正在构建节点 embedding 矩阵...")
        embeddings = []
        valid_indices = []
        for i, node in enumerate(self.nodes):
            emb = self._parse_embedding(node["properties"].get("embedding"))
            if emb is not None and emb.size > 0:
                embeddings.append(emb)
                valid_indices.append(i)
            else:
                print(f"⚠️ 节点 {node['id']} 缺少有效 embedding，跳过")

        if not embeddings:
            raise ValueError("未找到任何有效 embedding！请检查 graph_data.jsonl")

        self.node_embeddings = np.stack(embeddings)
        old_nodes = self.nodes
        self.nodes = [old_nodes[i] for i in valid_indices]
        self.node_id_to_index = {node["id"]: idx for idx, node in enumerate(self.nodes)}
        print(f"✅ 构建完成: {self.node_embeddings.shape[0]} 个有效节点, 维度 {self.node_embeddings.shape[1]}")

    def _init_embedding_model(self):
        # Lazy load: 第一次调用时才加载
        if not hasattr(self, '_model'):
            print("⏳ 首次加载 SentenceTransformer('BAAI/bge-multilingual-gemma2')...")
            from sentence_transformers import SentenceTransformer

             # 🔁 配置重试：最多 5 次，指数退避
            retry_kwargs = {
            "max_retries": 5,
            "backoff_factor": 1.5,
            }
            
            self._model = SentenceTransformer(
                "BAAI/bge-m3",
                model_kwargs={"torch_dtype": torch.float16},
                device="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
            )
            from huggingface_hub import configure_http_backend
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            retry_strategy = Retry(
                total=5,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            http = requests.Session()
            http.mount("https://", adapter)
            http.mount("http://", adapter)
            configure_http_backend(http)

            # 🔑 关键：设置统一的 prompt（与你当初生成 node embedding 时一致！）
            # 你当初构建 graph_data.jsonl 时用的什么 prompt？必须完全一致！
            # 根据你的样例，很可能是：
            self._instruction = "Given a scientific query about medical imaging, retrieve relevant papers, methods, or tasks."
            self._prompt = f'<instruct>{self._instruction}\n<query>'
            print(f"✅ 模型加载完成，prompt = '{self._prompt[:50]}...'")
    
    def _text_to_query_embedding(self, text: str) -> np.ndarray:
        self._init_embedding_model()  # 确保已加载
        # 🔑 使用 **相同 prompt** encode！
        embedding = self._model.encode(
            [text],  # 注意：encode 要求 list
            prompt=self._prompt,
            convert_to_numpy=True,
            normalize_embeddings=True  # 可选，但推荐（提升 cosine 相似度稳定性）
        )[0]
        return embedding.astype(np.float32)

    def retrieve_nodes_by_embedding(self, query: str, top_k: int = 5) -> List[Dict]:
        query_emb = self._text_to_query_embedding(query)
        if query_emb.shape[0] != self.node_embeddings.shape[1]:
            raise ValueError(
                f"维度不匹配！Query: {query_emb.shape[0]} ≠ Nodes: {self.node_embeddings.shape[1]}\n"
                "请确认：1) node embedding 也是用 bge-multilingual-gemma2 + 相同 prompt 生成；2) 没有后处理（如 PCA）"
            )
        sims = cosine_similarity([query_emb], self.node_embeddings)[0]
        top_idxs = np.argsort(sims)[-top_k:][::-1]
        results = []
        for i in top_idxs:
            if sims[i] > 0.2:  # 降低阈值，避免漏召
                results.append(self.nodes[i])
        return results
    
    def get_neighbors(self, node_id: str) -> List[Dict]:
        neighbors = []
        for rel in self.relationships:
            if rel["start_node_id"] == node_id:
                other_id = rel["end_node_id"]
            elif rel["end_node_id"] == node_id:
                other_id = rel["start_node_id"]
            else:
                continue
            idx = self.node_id_to_index.get(other_id)
            if idx is not None:
                neighbors.append({
                    "relationship": rel["label"],
                    "node": self.nodes[idx]
                })
        return neighbors

    def rag_context(self, question: str, top_k_nodes: int = 3) -> str:
        retrieved = self.retrieve_nodes_by_embedding(question, top_k=top_k_nodes)
        lines = []
        for node in retrieved:
            props = node["properties"]
            label = node["labels"][0] if node["labels"] else "Node"
            if label == "Paper":
                title = props.get("title", "N/A")
                year = props.get("year", "N/A")
                paper_id = props.get("paper_id", "N/A")
                lines.append(f"📄 论文: 《{title}》 (ID: {paper_id}, 年份: {year})")
                # 加一跳邻居
                for nb in self.get_neighbors(node["id"]):
                    nb_node = nb["node"]
                    nb_label = nb_node["labels"][0] if nb_node["labels"] else "Node"
                    nb_props = nb_node["properties"]
                    if nb_label == "Task":
                        lines.append(f"   → 研究任务: {nb_props.get('name', 'N/A')}")
                    elif nb_label == "Method":
                        lines.append(f"   → 提出方法: {nb_props.get('name', 'N/A')}")
                    elif nb_label == "ImagingModality":
                        lines.append(f"   → 使用模态: {nb_props.get('name', 'N/A')}")
            elif label in ["Task", "ImagingModality", "Method", "AnatomicalStructure"]:
                name = props.get("name", "N/A")
                lines.append(f"🏷️ {label}: {name} (ID: {props.get('id', 'N/A')})")
        return "\n".join(lines) if lines else "(无相关节点)"
    # ...（get_neighbors 和 rag_context 保持不变，略）
