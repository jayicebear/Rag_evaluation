import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np

# ===== 설정 =====
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "rag_test"
CHROMA_DB_PATH = "./test_chunk/chroma_db"  # 저장 경로 지정 (저장 코드와 동일하게)

# ===== ChromaDB 클라이언트 (저장된 DB 로드) =====
client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection(name=COLLECTION_NAME)

print(f"ChromaDB 로드: {CHROMA_DB_PATH}")
print(f" 컬렉션 문서 수: {collection.count()}")

# ===== 임베딩 모델 =====
model = SentenceTransformer(EMBED_MODEL)

# ===== BM25 인덱스 생성 =====
print("BM25 인덱스 생성 중...")

# ChromaDB에서 모든 문서 가져오기
all_docs = collection.get(include=["documents", "metadatas"])
documents = all_docs['documents']
ids = all_docs['ids']

# 토크나이징 (공백 기준)
tokenized_corpus = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

print(f" BM25 인덱스 생성 완료 ({len(documents)}개 문서)")

# ===== 검색 함수들 =====
def chroma_vector_search(query: str, top_k: int = 5) -> List[str]:
    """Vector Search (Embedding)"""
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    return results['ids'][0] if results['ids'] else []


def bm25_search(query: str, top_k: int = 5) -> List[str]:
    """BM25 Search (Keyword)"""
    # 쿼리 토크나이징
    tokenized_query = query.split()
    
    # BM25 점수 계산
    scores = bm25.get_scores(tokenized_query)
    
    # Top-K 선택
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    return [ids[i] for i in top_indices]


# ===== 평가 지표 계산 =====
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
    """전체 쿼리에 대한 Recall + MRR 평가"""
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    reciprocal_ranks = []
    total = 0
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {method_name}")
    print(f"{'='*80}")
    
    for item in queries:
        local_chunk_id = item['chunk_id']
        full_chunk_id = f"{document_name}_{local_chunk_id}"
        questions = item['question']
        
        for question in questions:
            # 검색 실행
            retrieved_ids = search_func(question, top_k=5)
            
            # Recall 계산
            recall_at_1 += calculate_recall(retrieved_ids, full_chunk_id, k=1)
            recall_at_3 += calculate_recall(retrieved_ids, full_chunk_id, k=3)
            recall_at_5 += calculate_recall(retrieved_ids, full_chunk_id, k=5)
            
            # MRR 계산
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
    
    print(f"\nResults for {method_name}:")
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
    file_name = 'raptor'
    json_file = f"./chunk_and_question/{file_name}_chunk_questions.json"
    
    print(f"Loading data from: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    queries = []
    for item in data:
        queries.append({
            "chunk_id": item["chunk_id"],
            "question": item["question"]
        })
    
    total_questions = sum(len(q["question"]) for q in queries)
    print(f"Loaded {len(queries)} chunks with {total_questions} total questions\n")
    
    results = []
    
    # 1. Vector Search (Embedding) 평가
    results.append(evaluate_metrics(
        queries, 
        chroma_vector_search, 
        "Vector Search", 
        file_name
    ))
    
    # 2. BM25 Search (Keyword) 평가
    results.append(evaluate_metrics(
        queries, 
        bm25_search, 
        "BM25 Search", 
        file_name
    ))
    
    # 최종 결과 비교
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS SUMMARY - {file_name}")
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