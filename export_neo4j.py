import json
from neo4j import GraphDatabase
from datetime import date, datetime

# ================= 配置区域 =================
URI = "neo4j+s://e96b056a.databases.neo4j.io"  # 替换你的 Aura 连接地址
AUTH = ("neo4j", "l_Xozo1gLym66VVmHMXa9WMNmpju9uUsScSXtYy-elc")                # 替换你的密码
OUTPUT_FILE = "graph_data.jsonl"                 # 导出文件名
# ===========================================

# 辅助函数：处理 Neo4j 特殊数据类型（如日期、大整数）
def custom_serializer(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)

def export_graph_streaming(uri, auth, filename):
    driver = GraphDatabase.driver(uri, auth=auth)
    
    try:
        driver.verify_connectivity()
        print(f"连接成功，准备导出到 {filename} ...")
        
        with open(filename, 'w', encoding='utf-8') as f:
            with driver.session() as session:
                
                # --- 第一步：导出所有节点 ---
                print("正在导出节点...", end="", flush=True)
                node_count = 0
                
                # 使用 iter() 避免一次性把所有数据加载到内存
                # 这里的查询只返回节点对象，不进行复杂的 APOC 处理
                result = session.run("MATCH (n) RETURN n")
                
                for record in result:
                    node = record["n"]
                    # 构造简单的字典格式
                    node_data = {
                        "type": "node",
                        "id": node.element_id,  # 或 node.id
                        "labels": list(node.labels),
                        "properties": dict(node)
                    }
                    # 写入一行 JSON
                    f.write(json.dumps(node_data, default=custom_serializer, ensure_ascii=False) + "\n")
                    node_count += 1
                    if node_count % 1000 == 0:
                        print(".", end="", flush=True)
                
                print(f"\n✅ 节点导出完成：{node_count} 个")

                # --- 第二步：导出所有关系 ---
                print("正在导出关系...", end="", flush=True)
                rel_count = 0
                
                # 查询关系，同时获取起始和结束节点的ID
                # 这样下游任务才能把关系连起来
                rel_query = """
                MATCH (start)-[r]->(end)
                RETURN r, start, end
                """
                result = session.run(rel_query)
                
                for record in result:
                    rel = record["r"]
                    start_node = record["start"]
                    end_node = record["end"]
                    
                    rel_data = {
                        "type": "relationship",
                        "id": rel.element_id,
                        "start_node_id": start_node.element_id,
                        "end_node_id": end_node.element_id,
                        "label": rel.type,
                        "properties": dict(rel)
                    }
                    f.write(json.dumps(rel_data, default=custom_serializer, ensure_ascii=False) + "\n")
                    rel_count += 1
                    if rel_count % 1000 == 0:
                        print(".", end="", flush=True)
                        
                print(f"\n✅ 关系导出完成：{rel_count} 个")
                
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        driver.close()
        print("连接已关闭。")

if __name__ == "__main__":
    export_graph_streaming(URI, AUTH, OUTPUT_FILE)
