from pymongo import MongoClient
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from urllib.parse import quote_plus
import json
from typing import List, Dict
from collections import defaultdict
from rank_bm25 import BM25Okapi
import numpy as np

# ===== 설정 =====
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# MongoDB 설정
user_name = 'jimin_lee'
password = quote_plus("Superjm09!")
#MONGO_URI = f"mongodb+srv://{user_name}:{password}@jaycluster.qvqia6n.mongodb.net/?retryWrites=true&w=majority&appName=jaycluster"
MONGO_URI=f"mongodb://zium:zium1207!!@192.168.0.43:27017/?authSource=admin"

MONGO_DB_NAME = "rag_test"
MONGO_COLLECTION_NAME = "chunks_for_test"

# ChromaDB 설정
CHROMA_DB_PATH = "./chroma/chroma_db"
CHROMA_COLLECTION_NAME = "rag_test"

# ===== MongoDB 연결 =====
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
mongo_collection = mongo_db[MONGO_COLLECTION_NAME]

# ===== ChromaDB 연결 =====
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)
chroma_collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)

# ===== 임베딩 모델 =====
model = SentenceTransformer(EMBED_MODEL)

print(f"✅ MongoDB 연결: {MONGO_DB_NAME}.{MONGO_COLLECTION_NAME}")
print(f"✅ ChromaDB 연결: {CHROMA_DB_PATH}/{CHROMA_COLLECTION_NAME}")
print(f"📊 ChromaDB 문서 수: {chroma_collection.count()}")

# ===== BM25 인덱스 생성 (ChromaDB용) =====
print("🔨 BM25 인덱스 생성 중...")
all_docs = chroma_collection.get(include=["documents"])
bm25_documents = all_docs['documents']
bm25_ids = all_docs['ids']
tokenized_corpus = [doc.split() for doc in bm25_documents]
bm25 = BM25Okapi(tokenized_corpus)
print(f"✅ BM25 인덱스 생성 완료 ({len(bm25_documents)}개 문서)")

# ===== RRF 상수 =====
RRF_K = 60

# ===== MongoDB 검색 함수들 =====
def mongo_vector_search(query, top_k=5):
    """MongoDB Vector Search"""
    query_embedding = model.encode(query).tolist()
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "chunk_embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": top_k
            }
        },
        {
            "$project": {
                "chunk_id": 1,
                "_id": 0
            }
        }
    ]
    
    results = list(mongo_collection.aggregate(pipeline))
    return [r['chunk_id'] for r in results]


def mongo_text_search(query, top_k=5):
    """MongoDB Text Search"""
    pipeline = [
        {
            "$search": {
                "index": "text_index",
                "text": {
                    "query": query,
                    "path": "chunk"
                }
            }
        },
        {"$limit": top_k},
        {
            "$project": {
                "chunk_id": 1,
                "_id": 0
            }
        }
    ]
    
    results = list(mongo_collection.aggregate(pipeline))
    return [r['chunk_id'] for r in results]


def mongo_hybrid_search(query, top_k=5):
    """MongoDB Hybrid Search (Manual RRF)"""
    # Vector
    query_embedding = model.encode(query).tolist()
    vector_pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "chunk_embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": top_k * 2
            }
        },
        {"$project": {"chunk_id": 1, "_id": 0}}
    ]
    vector_results = [(r['chunk_id'], idx) for idx, r in enumerate(mongo_collection.aggregate(vector_pipeline))]
    
    # Text
    text_pipeline = [
        {
            "$search": {
                "index": "text_index",
                "text": {"query": query, "path": "chunk"}
            }
        },
        {"$limit": top_k * 2},
        {"$project": {"chunk_id": 1, "_id": 0}}
    ]
    text_results = [(r['chunk_id'], idx) for idx, r in enumerate(mongo_collection.aggregate(text_pipeline))]
    
    # RRF
    rrf_scores = defaultdict(float)
    for chunk_id, rank in vector_results:
        rrf_scores[chunk_id] += 1.0 / (RRF_K + rank + 1)
    for chunk_id, rank in text_results:
        rrf_scores[chunk_id] += 1.0 / (RRF_K + rank + 1)
    
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in sorted_results[:top_k]]


# ===== ChromaDB 검색 함수들 =====
def chroma_vector_search(query, top_k=5):
    """ChromaDB Vector Search"""
    query_embedding = model.encode(query).tolist()
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['ids'][0] if results['ids'] else []


def chroma_bm25_search(query, top_k=5):
    """ChromaDB BM25 Search"""
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [bm25_ids[i] for i in top_indices]

def chroma_hybrid_search(query, top_k=5):
    """ChromaDB Hybrid Search (Manual RRF)"""
    # Vector Search
    query_embedding = model.encode(query).tolist()
    vector_results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2
    )
    vector_ids = vector_results['ids'][0] if vector_results['ids'] else []

    # BM25 Search
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k * 2]
    bm25_ids_top = [bm25_ids[i] for i in top_indices]

    # RRF Fusion
    rrf_scores = defaultdict(float)
    for rank, doc_id in enumerate(vector_ids):
        rrf_scores[doc_id] += 1.0 / (RRF_K + rank + 1)
    for rank, doc_id in enumerate(bm25_ids_top):
        rrf_scores[doc_id] += 1.0 / (RRF_K + rank + 1)

    # Sort by fused score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    fused_ids = [doc_id for doc_id, _ in sorted_results[:top_k]]
    return fused_ids
# ===== 평가 함수 =====
def calculate_recall(retrieved_ids: List[str], ground_truth_id: str, k: int) -> int:
    """Recall@k 계산"""
    top_k_ids = retrieved_ids[:k]
    return 1 if ground_truth_id in top_k_ids else 0


def calculate_reciprocal_rank(retrieved_ids: List[str], ground_truth_id: str) -> float:
    """MRR 계산"""
    try:
        rank = retrieved_ids.index(ground_truth_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def evaluate_metrics(queries: List[Dict], search_func, method_name: str, document_name: str):
    """전체 쿼리 평가"""
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    reciprocal_ranks = []
    total = 0
    
    print(f"\n{'='*80}")
    print(f"📊 Evaluating: {method_name}")
    print(f"{'='*80}")
    
    for item in queries:
        local_chunk_id = item['chunk_id']
        full_chunk_id = f"{document_name}_{local_chunk_id}"
        questions = item['question']
        
        for question in questions:
            retrieved_ids = search_func(question, top_k=5)
            
            recall_at_1 += calculate_recall(retrieved_ids, full_chunk_id, k=1)
            recall_at_3 += calculate_recall(retrieved_ids, full_chunk_id, k=3)
            recall_at_5 += calculate_recall(retrieved_ids, full_chunk_id, k=5)
            
            rr = calculate_reciprocal_rank(retrieved_ids, full_chunk_id)
            reciprocal_ranks.append(rr)
            
            total += 1
            
            if total % 10 == 0:
                print(f"  Processed {total} queries...")
    
    recall_1 = (recall_at_1 / total) * 100
    recall_3 = (recall_at_3 / total) * 100
    recall_5 = (recall_at_5 / total) * 100
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    print(f"\n📈 Results for {method_name}:")
    print(f"  Total Queries: {total}")
    print(f"  Recall@1: {recall_1:.2f}%")
    print(f"  Recall@3: {recall_3:.2f}%")
    print(f"  Recall@5: {recall_5:.2f}%")
    print(f"  MRR: {mrr:.4f}")
    
    return {
        "method": method_name,
        "total": total,
        "recall@1": recall_1,
        "recall@3": recall_3,
        "recall@5": recall_5,
        "mrr": mrr
    }


# ===== 메인 실행 =====
if __name__ == "__main__":
    file_name = 'short_story'
    json_file = f"./chunk_and_question/{file_name}_chunk_questions.json"
    
    print(f"\n📂 Loading data from: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    queries = []
    for item in data:
        queries.append({
            "chunk_id": item["chunk_id"],
            "question": item["question"]
        })
    
    total_questions = sum(len(q["question"]) for q in queries)
    print(f"✅ Loaded {len(queries)} chunks with {total_questions} total questions\n")
    
    results = []
    
    # MongoDB 평가
    print("\n" + "="*80)
    print("🔵 MONGODB EVALUATION")
    print("="*80)
    results.append(evaluate_metrics(queries, mongo_vector_search, "MongoDB - Vector", file_name))
    results.append(evaluate_metrics(queries, mongo_text_search, "MongoDB - Text", file_name))
    results.append(evaluate_metrics(queries, mongo_hybrid_search, "MongoDB - Hybrid", file_name))
    
    # ChromaDB 평가
    print("\n" + "="*80)
    print("🟢 CHROMADB EVALUATION")
    print("="*80)
    results.append(evaluate_metrics(queries, chroma_vector_search, "ChromaDB - Vector", file_name))
    results.append(evaluate_metrics(queries, chroma_bm25_search, "ChromaDB - BM25", file_name))
    results.append(evaluate_metrics(queries, chroma_hybrid_search, "ChromaDB - Hybrid", file_name))

    # 최종 결과 비교
    print(f"\n{'='*80}")
    print(f"📊 FINAL RESULTS COMPARISON - {file_name}")
    print(f"{'='*80}")
    print(f"{'Method':<35} {'Recall@1':<12} {'Recall@3':<12} {'Recall@5':<12} {'MRR':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['method']:<35} "
              f"{result['recall@1']:>10.2f}% "
              f"{result['recall@3']:>10.2f}% "
              f"{result['recall@5']:>10.2f}% "
              f"{result['mrr']:>8.4f}")
    
    print("=" * 80)
    import json
    output_file = f"./evaluation_results/{file_name}_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)