from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from urllib.parse import quote_plus
import json
from typing import List, Dict
from collections import defaultdict

# ===== 설정 =====
user_name = 'jimin_lee'
password = quote_plus("Superjm09!")
MONGO_URI = f"mongodb+srv://{user_name}:{password}@jaycluster.qvqia6n.mongodb.net/?retryWrites=true&w=majority&appName=jaycluster"

DB_NAME = "gnos_ai_base"
COLLECTION_NAME = "chunk_with_questions"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# ===== 연결 =====
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
model = SentenceTransformer(EMBED_MODEL)

# ===== RRF 상수 =====
RRF_K = 60

# ===== 검색 함수들 =====
def mongo_vector_search_with_rank(query, top_k=10):
    """Vector Search - chunk_id와 순위 반환"""
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
                "score": {"$meta": "vectorSearchScore"},
                "_id": 0
            }
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    return [(r['chunk_id'], idx) for idx, r in enumerate(results)]


def mongo_text_search_with_rank(query, top_k=10):
    """Text Search - chunk_id와 순위 반환"""
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
                "score": {"$meta": "searchScore"},
                "_id": 0
            }
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    return [(r['chunk_id'], idx) for idx, r in enumerate(results)]


def reciprocal_rank_fusion(vector_results: List[tuple], text_results: List[tuple], k: int = RRF_K) -> List[str]:
    """Reciprocal Rank Fusion (RRF) 계산"""
    rrf_scores = defaultdict(float)
    
    for chunk_id, rank in vector_results:
        rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
    
    for chunk_id, rank in text_results:
        rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
    
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in sorted_results]


def mongo_hybrid_search(query, top_k=5):
    """Hybrid Search with Manual RRF"""
    vector_results = mongo_vector_search_with_rank(query, top_k=top_k * 2)
    text_results = mongo_text_search_with_rank(query, top_k=top_k * 2)
    hybrid_results = reciprocal_rank_fusion(vector_results, text_results)
    return hybrid_results[:top_k]


def mongo_vector_search(query, top_k=5):
    """Vector Search만"""
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
    
    results = list(collection.aggregate(pipeline))
    return [r['chunk_id'] for r in results]


def mongo_text_search(query, top_k=5):
    """Text Search만"""
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
    
    results = list(collection.aggregate(pipeline))
    return [r['chunk_id'] for r in results]


# ===== 평가 지표 계산 함수 =====
def calculate_recall(retrieved_ids: List[str], ground_truth_id: str, k: int) -> int:
    """
    Recall@k 계산
    
    Args:
        retrieved_ids: 검색된 chunk_id 리스트 (문자열)
        ground_truth_id: 정답 chunk_id (문자열)
        k: top-k
    """
    top_k_ids = retrieved_ids[:k]
    return 1 if ground_truth_id in top_k_ids else 0


def calculate_reciprocal_rank(retrieved_ids: List[str], ground_truth_id: str) -> float:
    """
    MRR (Mean Reciprocal Rank) 계산
    
    Args:
        retrieved_ids: 검색된 chunk_id 리스트 (문자열)
        ground_truth_id: 정답 chunk_id (문자열)
    
    Returns:
        Reciprocal Rank (1/rank if found, 0 if not found)
    """
    try:
        rank = retrieved_ids.index(ground_truth_id) + 1  # 1-based index
        return 1.0 / rank
    except ValueError:
        return 0.0  # 정답이 검색 결과에 없음


def evaluate_metrics(queries: List[Dict], search_func, method_name: str, document_name: str):
    """
    전체 쿼리에 대한 Recall + MRR 평가
    
    Args:
        queries: 쿼리 리스트
        search_func: 검색 함수
        method_name: 검색 방법 이름
        document_name: 문서 이름 (chunk_id 생성용)
    """
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    reciprocal_ranks = []
    total = 0
    
    print(f"\n{'='*80}")
    print(f"📊 Evaluating: {method_name}")
    print(f"{'='*80}")
    
    for item in queries:
        local_chunk_id = item['chunk_id']  # JSON의 원래 chunk_id (0, 1, 2...)
        
        # ✅ 복합 chunk_id 생성
        full_chunk_id = f"{document_name}_{local_chunk_id}"
        
        questions = item['question']
        
        for question in questions:
            # 검색 실행
            retrieved_ids = search_func(question, top_k=5)
            
            # Recall 계산 (문자열 비교)
            recall_at_1 += calculate_recall(retrieved_ids, full_chunk_id, k=1)
            recall_at_3 += calculate_recall(retrieved_ids, full_chunk_id, k=3)
            recall_at_5 += calculate_recall(retrieved_ids, full_chunk_id, k=5)
            
            # Reciprocal Rank 계산
            rr = calculate_reciprocal_rank(retrieved_ids, full_chunk_id)
            reciprocal_ranks.append(rr)
            
            total += 1
            
            if total % 10 == 0:
                print(f"  Processed {total} queries...")
    
    # 평균 계산
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
    # ✅ 평가할 문서 이름
    file_name = '자동차보험'
    
    json_file = f"/home/ljm/test_chunk/chunk_and_question/{file_name}_chunk_questions.json"
    
    print(f"📂 Loading data from: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    queries = []
    for item in data:
        queries.append({
            "chunk_id": item["chunk_id"],  # 원래 chunk_id (0, 1, 2...)
            "question": item["question"]
        })
    
    total_questions = sum(len(q["question"]) for q in queries)
    print(f"✅ Loaded {len(queries)} chunks with {total_questions} total questions\n")
    
    results = []
    
    # ✅ 평가 시 document_name 전달
    results.append(evaluate_metrics(queries, mongo_vector_search, "Vector Search", file_name))
    results.append(evaluate_metrics(queries, mongo_text_search, "Text Search", file_name))
    results.append(evaluate_metrics(queries, mongo_hybrid_search, "Hybrid Search (Manual RRF)", file_name))
    
    # 최종 결과
    print(f"\n{'='*80}")
    print(f"📊 FINAL RESULTS SUMMARY - {file_name}")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'Recall@1':<12} {'Recall@3':<12} {'Recall@5':<12} {'MRR':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['method']:<30} "
              f"{result['recall@1']:>10.2f}% "
              f"{result['recall@3']:>10.2f}% "
              f"{result['recall@5']:>10.2f}% "
              f"{result['mrr']:>8.4f}")
    
    print("=" * 80)