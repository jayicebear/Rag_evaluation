from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from tqdm import tqdm

# ===== 설정 =====
#MONGO_URI = "mongodb://zium:zium1207%21%21@192.168.0.43:27017/admin"

from urllib.parse import quote_plus

# ===== 설정 =====
user_name = 'jimin_lee'
password = quote_plus("Superjm09!")
MONGO_URI = f"mongodb+srv://{user_name}:{password}@jaycluster.qvqia6n.mongodb.net/?retryWrites=true&w=majority&appName=jaycluster"

DB_NAME = "gnos_ai_base"
COLLECTION_NAME = "chunk_with_questions"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_JSON_PATH = "./chunk_and_question/safe_sql_chunk_questions.json"

# ===== Mongo 연결 =====
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# ===== 모델 로드 =====
model = SentenceTransformer(EMBED_MODEL)

# ===== 질문 데이터 로드 =====
with open(CHUNK_JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} chunks with questions\n")

# ===== 개별 검색 함수 =====
def mongo_vector_search(query_emb, top_k=5):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vectorSearch",
                "path": "chunk_embedding",
                "queryVector": query_emb,
                "numCandidates": 100,
                "limit": top_k
            }
        },
                   {
            "$project": {
                "chunk_id": 1,  # ✅ 이게 핵심!
                "chunk": 1,
                "score": {"$meta": "vectorSearchScore"},  # 유사도 점수도 확인
                "_id": 0
            }
            
        }
    ]
    results = list(collection.aggregate(pipeline))
    print(results)
    return [r["chunk_id"] for r in results]

def mongo_text_search(query_text, top_k=5):
    pipeline = [
        {
            "$search": {
                "index": "search_idx",
                "text": {"query": query_text, "path": "chunk"}
            }
        },
        {"$limit": top_k},
        {"$project": {"chunk_id": 1, "_id": 0}}
    ]
    results = list(collection.aggregate(pipeline))
    #print(results)
    return [r["chunk_id"] for r in results]

def mongo_hybrid_rankfusion(query_text, query_emb, top_k=5, num_candidates=100, per_stage_limit=20):
    """MongoDB 8.2 $rankFusion 기반 Hybrid Search"""
    pipeline = [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vector": [
                            {
                                "$vectorSearch": {
                                    "index": "vector_idx_qwen3",
                                    "path": "chunk_embedding",
                                    "queryVector": query_emb,
                                    "numCandidates": num_candidates,
                                    "limit": per_stage_limit
                                }
                            },
                            {"$limit": per_stage_limit}
                        ],
                        "text": [
                            {
                                "$search": {
                                    "index": "search_idx",
                                    "text": {"query": query_text, "path": "chunk"}
                                }
                            },
                            {"$limit": per_stage_limit}
                        ]
                    }
                },
                "combination": {
                    "weights": {"vector": 0.7, "text": 0.3}
                },
                "scoreDetails": True
            }
        },
        {"$addFields": {"rrfScore": {"$meta": "score"}}},
        {"$sort": {"rrfScore": -1}},
        {"$limit": top_k},
        {"$project": {"chunk_id": 1, "rrfScore": 1, "_id": 0}}
    ]

    results = list(collection.aggregate(pipeline))
    #print(results)
    return [r["chunk_id"] for r in results]


# ===== 평가 함수 =====
def evaluate(method, k_list=[1, 3, 5]):
    recall = {k: 0 for k in k_list}
    total_q = 0

    for chunk in tqdm(data, desc=f"Evaluating {method}"):
        gt_id = chunk["chunk_id"]
        for q in chunk["question"]:
            total_q += 1
            q_emb = model.encode(q).tolist()

            if method == "vector":
                top_ids = mongo_vector_search(q_emb, max(k_list))
            elif method == "text":
                top_ids = mongo_text_search(q, max(k_list))
            elif method == "hybrid":
                top_ids = mongo_hybrid_rankfusion(q, q_emb, max(k_list))
            else:
                raise ValueError("Invalid method")

            for k in k_list:
                if gt_id in top_ids[:k]:
                    recall[k] += 1

    results = {f"Recall@{k}": recall[k] / total_q for k in k_list}
    return results


# ===== 실행 =====
methods = ["vector", "text", "hybrid"]

for m in methods:
    res = evaluate(m)
    print(f"\n📊 {m.upper()} SEARCH RESULTS:")
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")
