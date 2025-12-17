# re_embed_with_bge_m3.py
from sentence_transformers import SentenceTransformer
import json
import torch

print("📥 正在加载 BAAI/bge-m3 ...")
model = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
print("✅ 模型加载完成")

def get_text_for_node(node):
    props = node["properties"]
    labels = node["labels"]
    typ = labels[0] if labels else "Node"
    if typ == "Paper":
        return f"{props.get('title', '')} {props.get('category', '')} {props.get('authors', '')}"
    elif typ in ["Task", "ImagingModality", "AnatomicalStructure", "Method", "Dataset", "Metric"]:
        return props.get("name", "") or props.get("description", "")
    elif typ == "Innovation":
        return props.get("description", "")
    else:
        return str(props)

input_path = "/home/shijc/knowledgegraph-main/graph_data.jsonl"
output_path = "/home/shijc/knowledgegraph-main/graph_data_bge-m3.jsonl"

print(f"🔄 正在重嵌入 {input_path} → {output_path} ...")

with open(output_path, "w", encoding="utf-8") as fout:
    with open(input_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin, 1):
            obj = json.loads(line)
            if obj["type"] == "node":
                text = get_text_for_node(obj)
                emb = model.encode([text], normalize_embeddings=True)[0].tolist()
                obj["properties"]["embedding"] = emb  # 覆盖旧 embedding
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"  已处理 {i} 行")

print("🎉 完成！新文件已保存至:", output_path)
