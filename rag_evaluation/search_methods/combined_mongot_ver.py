import os
import json
import argparse
import numpy as np
from typing import List, Dict
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings
from urllib.parse import quote_plus

# ===============================
# argparse 설정
# ===============================
parser = argparse.ArgumentParser(description="Evaluate RAG search performance across MongoDB and ChromaDB")
parser.add_argument("--file_name", type=str, required=True, help="평가할 파일 이름 (예: safe_sql, raptor 등)")
parser.add_argument("--mode", type=str, choices=["original", "rewrite","passage"], default="rewrite",
                    help="사용할 질문 JSON 유형 (original=chunk_questions, rewritten=rewritten_query)")
args = parser.parse_args()

file_name = args.file_name
input_type = args.mode

# ===============================
# 설정
# ===============================
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# MongoDB 설정
user_name = 'jimin_lee'
password = quote_plus("Superjm09!")
MONGO_URI = f"mongodb://zium:zium1207%21%21@192.168.0.43:27017/admin"
MONGO_DB_NAME = "rag_test"
MONGO_COLLECTION_NAME = "chunks_for_test"

# ChromaDB 설정
CHROMA_DB_PATH = "./chroma/chroma_db"
CHROMA_COLLECTION_NAME = "rag_test"

# ===============================
# DB 연결
# ===============================
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
mongo_collection = mongo_db[MONGO_COLLECTION_NAME]

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)
chroma_collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)

# 임베딩 모델
model = SentenceTransformer(EMBED_MODEL)

print(f"✅ MongoDB 연결: {MONGO_DB_NAME}.{MONGO_COLLECTION_NAME}")
print(f"✅ ChromaDB 연결: {CHROMA_DB_PATH}/{CHROMA_COLLECTION_NAME}")
print(f"📊 MongoDB 문서 수: {mongo_collection.count_documents({})}")
print(f"📊 ChromaDB 문서 수: {chroma_collection.count()}")

# ===============================
# BM25 인덱스 생성
# ===============================
print("🔨 BM25 인덱스 생성 중...")
all_docs = chroma_collection.get(include=["documents"])
bm25_documents = all_docs['documents']
bm25_ids = all_docs['ids']
tokenized_corpus = [doc.split() for doc in bm25_documents]
bm25 = BM25Okapi(tokenized_corpus)
print(f"✅ BM25 인덱스 생성 완료 ({len(bm25_documents)}개 문서)")

RRF_K = 60


# ===============================
# Chunk ID 정규화
# ===============================
def normalize_chunk_id(chunk_id: str) -> str:
    if '_chunk_questions_' in chunk_id:
        parts = chunk_id.split('_chunk_questions_')
        if len(parts) == 2:
            return f"{parts[0]}_{parts[1]}"
    return chunk_id


# ===============================
# MongoDB 검색 함수들
# ===============================
def mongo_vector_search(query, top_k=5):
    query_embedding = model.encode(query).tolist()
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_search",
                "path": "chunk_embedding",
                "queryVector": query_embedding,
                "exact": True,
                "limit": top_k
            }
        },
        {"$project": {"chunk_id": 1, "_id": 0}}
    ]
    results = list(mongo_collection.aggregate(pipeline))
    return [normalize_chunk_id(r['chunk_id']) for r in results]


def mongo_text_search(query, top_k=5):
    pipeline = [
        {
            "$search": {
                "index": "text_search",
                "text": {"query": query, "path": "chunk"}
            }
        },
        {"$limit": top_k},
        {"$project": {"chunk_id": 1, "_id": 0}}
    ]
    results = list(mongo_collection.aggregate(pipeline))
    return [normalize_chunk_id(r['chunk_id']) for r in results]


def mongo_hybrid_search(query, top_k=5):
    query_embedding = model.encode(query).tolist()
    vector_pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_search",
                "path": "chunk_embedding",
                "queryVector": query_embedding,
                "exact": True,
                "limit": top_k * 2
            }
        },
        {"$project": {"chunk_id": 1, "_id": 0}}
    ]
    text_pipeline = [
        {
            "$search": {
                "index": "text_search",
                "text": {"query": query, "path": "chunk"}
            }
        },
        {"$limit": top_k * 2},
        {"$project": {"chunk_id": 1, "_id": 0}}
    ]
    vector_results = [(normalize_chunk_id(r['chunk_id']), idx) for idx, r in enumerate(mongo_collection.aggregate(vector_pipeline))]
    text_results = [(normalize_chunk_id(r['chunk_id']), idx) for idx, r in enumerate(mongo_collection.aggregate(text_pipeline))]

    rrf_scores = defaultdict(float)
    for cid, rank in vector_results:
        rrf_scores[cid] += 1.0 / (RRF_K + rank + 1)
    for cid, rank in text_results:
        rrf_scores[cid] += 1.0 / (RRF_K + rank + 1)
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in sorted_results[:top_k]]


# ===============================
# ChromaDB 검색 함수들
# ===============================
def chroma_vector_search(query, top_k=5):
    query_embedding = model.encode(query).tolist()
    results = chroma_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['ids'][0] if results['ids'] else []


def chroma_bm25_search(query, top_k=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [bm25_ids[i] for i in top_indices]


def chroma_hybrid_search(query, top_k=5):
    query_embedding = model.encode(query).tolist()
    vector_results = chroma_collection.query(query_embeddings=[query_embedding], n_results=top_k * 2)
    vector_ids = vector_results['ids'][0] if vector_results['ids'] else []

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k * 2]
    bm25_ids_top = [bm25_ids[i] for i in top_indices]

    rrf_scores = defaultdict(float)
    for rank, doc_id in enumerate(vector_ids):
        rrf_scores[doc_id] += 1.0 / (RRF_K + rank + 1)
    for rank, doc_id in enumerate(bm25_ids_top):
        rrf_scores[doc_id] += 1.0 / (RRF_K + rank + 1)
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_results[:top_k]]


# ===============================
# 평가 함수들
# ===============================
def calculate_recall(retrieved_ids: List[str], ground_truth_id: str, k: int) -> int:
    return 1 if ground_truth_id in retrieved_ids[:k] else 0


def calculate_reciprocal_rank(retrieved_ids: List[str], ground_truth_id: str) -> float:
    try:
        return 1.0 / (retrieved_ids.index(ground_truth_id) + 1)
    except ValueError:
        return 0.0


def evaluate_metrics(queries: List[Dict], search_func, method_name: str, document_name: str):
    recall_at_1 = recall_at_3 = recall_at_5 = 0
    reciprocal_ranks = []
    total = 0
    print(f"\n📊 Evaluating: {method_name}")
    for item in queries:
        full_chunk_id = f"{document_name}_{item['chunk_id']}"
        for question in item["question"]:
            retrieved_ids = search_func(question, top_k=5)
            recall_at_1 += calculate_recall(retrieved_ids, full_chunk_id, 1)
            recall_at_3 += calculate_recall(retrieved_ids, full_chunk_id, 3)
            recall_at_5 += calculate_recall(retrieved_ids, full_chunk_id, 5)
            reciprocal_ranks.append(calculate_reciprocal_rank(retrieved_ids, full_chunk_id))
            total += 1

    recall_1 = recall_at_1 / total * 100
    recall_3 = recall_at_3 / total * 100
    recall_5 = recall_at_5 / total * 100
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"✅ {method_name}: R@1={recall_1:.2f}%, R@3={recall_3:.2f}%, R@5={recall_5:.2f}%, MRR={mrr:.4f}")
    return {"method": method_name, "total": total, "recall@1": recall_1, "recall@3": recall_3, "recall@5": recall_5, "mrr": mrr}


# ===============================
# 메인 실행
# ===============================
if __name__ == "__main__":
    if input_type == "original":
        json_file = f"./chunk_and_question/{file_name}_chunk_questions.json"
    else:
        json_file = f"./rewritten_query/{file_name}_{input_type}_rewritten.json"


    print(f"\n📂 Loading data from: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = [{"chunk_id": item["chunk_id"], "question": item["question"]} for item in data]
    total_questions = sum(len(q["question"]) for q in queries)
    print(f"✅ Loaded {len(queries)} chunks with {total_questions} total questions")

    results = []
    results.append(evaluate_metrics(queries, mongo_vector_search, "MongoDB - Vector", file_name))
    results.append(evaluate_metrics(queries, mongo_text_search, "MongoDB - Text", file_name))
    results.append(evaluate_metrics(queries, mongo_hybrid_search, "MongoDB - Hybrid", file_name))
    results.append(evaluate_metrics(queries, chroma_vector_search, "ChromaDB - Vector", file_name))
    results.append(evaluate_metrics(queries, chroma_bm25_search, "ChromaDB - BM25", file_name))
    results.append(evaluate_metrics(queries, chroma_hybrid_search, "ChromaDB - Hybrid", file_name))
    
    print(f"\n{'='*80}")
    print(f"📊 FINAL RESULTS COMPARISON - {file_name} ({input_type})")
    print(f"{'='*80}")
    print(f"{'Method':<35} {'Recall@1':<12} {'Recall@3':<12} {'Recall@5':<12} {'MRR':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['method']:<35} "
              f"{r['recall@1']:>10.2f}% "
              f"{r['recall@3']:>10.2f}% "
              f"{r['recall@5']:>10.2f}% "
              f"{r['mrr']:>8.4f}")
    print("=" * 80)
    output_file = f"./evaluation_results/{file_name}_evaluation_results_{input_type}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n💾 Results saved to: {output_file}")
