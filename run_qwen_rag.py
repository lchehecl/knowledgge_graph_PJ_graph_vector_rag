# run_qwen_rag.py
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # ← 关键！只暴露 GPU 1 给 PyTorch
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免 tokenizer 多线程冲突
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 可选：清空已有缓存（保险起见）
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
from dashscope import Generation
from tqdm import tqdm
import time
from graph_rag import TmiGraphRAG

# ================= 配置 =================
DASHSCOPE_API_KEY = "sk-"  # ← 替换为你的阿里云 API Key！
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

MODEL_NAME = "qwen-turbo"  # 可选：qwen-turbo, qwen-max, qwen-plus
JSONL_PATH = "/home/shijc/knowledgegraph-main/graph_data_bge-m3.jsonl"

# 初始化图 RAG
graph_rag = TmiGraphRAG(JSONL_PATH)

def call_qwen(prompt: str, model: str = MODEL_NAME) -> str:
    try:
        response = Generation.call(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        if response.status_code == 200:
            return response.output["text"].strip()
        else:
            raise RuntimeError(f"Qwen API error: {response.code} - {response.message}")
    except Exception as e:
        print(f"⚠️ Qwen 调用失败: {e}")
        return ""

# ================= 模拟一个问题 =================
question = "有哪些论文使用了 MRI 模态来研究脑肿瘤分割？"

print("🔍 问题:", question)
print("=" * 60)

# Vanilla 回答（无检索）
print("\n🤖 Vanilla Qwen 回答（无知识库）:")
vanilla_ans = call_qwen(question)
print(vanilla_ans)

# RAG 回答
print("\n📚 RAG 增强回答（检索知识图谱后）:")
context = graph_rag.rag_context(question, top_k_nodes=5)
print("🔍 检索到的上下文:")
print(context)
print("-" * 60)

rag_prompt = f"""
你是一位医学影像人工智能领域的专家，熟悉 TMI（IEEE Transactions on Medical Imaging）等顶刊论文。

请根据以下检索到的相关知识，回答用户的问题。
- 若知识相关，请基于知识作答，引用论文标题、方法名等具体信息。
- 若知识不相关或为空，请基于你自身的医学影像知识作答。
- 回答应专业、简洁、有依据。

【检索知识】
{context}

【问题】
{question}
"""

rag_ans = call_qwen(rag_prompt)
print("✅ RAG 回答:")
print(rag_ans)

# 可选：保存结果
import pandas as pd
df = pd.DataFrame([{
    "question": question,
    "vanilla": vanilla_ans,
    "rag_context": context,
    "rag_answer": rag_ans
}])
df.to_csv("qwen_rag_demo_output.csv", index=False, encoding="utf-8-sig")
print("\n💾 结果已保存至 qwen_rag_demo_output.csv") 
