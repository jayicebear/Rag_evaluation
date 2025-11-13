from pymongo import MongoClient
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from urllib.parse import quote_plus
import json
from typing import List, Dict
from collections import defaultdict
from rank_bm25 import BM25Okapi
import numpy as np
from reranker import QwenReranker
import torch 

# ===== 설정 =====
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"


# MongoDB 설정
user_name = 'jimin_lee'
password = quote_plus("Superjm09!")
#MONGO_URI = f"mongodb+srv://{user_name}:{password}@jaycluster.qvqia6n.mongodb.net/?retryWrites=true&w=majority&appName=jaycluster"
MONGO_URI=f"mongodb://zium:zium1207!!@192.168.0.43:27017/?authSource=admin"
MONGO_DB_NAME = "gnos_ai_base"
MONGO_COLLECTION_NAME = "chunk_with_questions"

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

# ===== 모델 로딩 =====
print("🔄 모델 로딩 중...")
model = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)
# if reranker.tokenizer.pad_token is None:
#     reranker.tokenizer.pad_token = reranker.tokenizer.eos_token
#     reranker.tokenizer.pad_token_id = reranker.tokenizer.eos_token_id
print("✅ 임베딩 모델 로딩 완료")
print("✅ Reranker 모델 로딩 완료")

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

# ===== 문서 텍스트 가져오기 함수 =====
def get_document_texts(chunk_ids: List[str]) -> Dict[str, str]:
    """chunk_id 리스트로부터 실제 문서 텍스트 가져오기"""
    # 중복 제거
    unique_chunk_ids = list(set(chunk_ids))
    
    if not unique_chunk_ids:
        return {}
    
    # ChromaDB에서 가져오기
    results = chroma_collection.get(
        ids=unique_chunk_ids,
        include=["documents"]
    )
    
    doc_map = {}
    if results['ids']:
        for chunk_id, doc in zip(results['ids'], results['documents']):
            doc_map[chunk_id] = doc
    
    return doc_map


def rerank_results(query: str, chunk_ids: List[str], top_k: int = 5, batch_size: int = 1) -> List[str]:
    """Reranker를 사용하여 결과 재정렬 (CrossEncoder.predict 기반)"""
    if not chunk_ids:
        return []

    # 중복 제거하면서 순서 유지
    seen = set()
    unique_chunk_ids = []
    for chunk_id in chunk_ids:
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique_chunk_ids.append(chunk_id)

    # 문서 텍스트 가져오기
    doc_texts = get_document_texts(unique_chunk_ids)

    # query-document 쌍 생성
    pairs = []
    valid_ids = []
    for chunk_id in unique_chunk_ids:
        if chunk_id in doc_texts:
            pairs.append([query, doc_texts[chunk_id]])
            valid_ids.append(chunk_id)

    if not pairs:
        return unique_chunk_ids[:top_k]

    # 배치로 Reranking 점수 계산
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            try:
                batch_scores = reranker.predict(batch_pairs)
                all_scores.extend(batch_scores)
            except Exception as e:
                print(f"⚠️ 배치 처리 실패, 개별 처리로 전환: {e}")
                for pair in batch_pairs:
                    score = reranker.predict([pair])[0]
                    all_scores.append(score)

            # GPU 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 점수로 정렬
    scored_results = list(zip(valid_ids, all_scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)

    return [chunk_id for chunk_id, _ in scored_results[:top_k]]

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


def mongo_vector_search_with_rerank(query, top_k=5, rerank_top_k=20):
    """MongoDB Vector Search + Reranking"""
    # 더 많은 후보 검색
    query_embedding = model.encode(query).tolist()
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "chunk_embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": rerank_top_k
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
    candidate_ids = [r['chunk_id'] for r in results]
    
    # Reranking
    return rerank_results(query, candidate_ids, top_k)


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


def mongo_text_search_with_rerank(query, top_k=5, rerank_top_k=20):
    """MongoDB Text Search + Reranking"""
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
        {"$limit": rerank_top_k},
        {
            "$project": {
                "chunk_id": 1,
                "_id": 0
            }
        }
    ]
    
    results = list(mongo_collection.aggregate(pipeline))
    candidate_ids = [r['chunk_id'] for r in results]
    
    # Reranking
    return rerank_results(query, candidate_ids, top_k)


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


def mongo_hybrid_search_with_rerank(query, top_k=5, rerank_top_k=20):
    """MongoDB Hybrid Search + Reranking"""
    # Vector
    query_embedding = model.encode(query).tolist()
    vector_pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "chunk_embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": rerank_top_k
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
        {"$limit": rerank_top_k},
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
    candidate_ids = [chunk_id for chunk_id, _ in sorted_results[:rerank_top_k]]
    
    # Reranking
    return rerank_results(query, candidate_ids, top_k)


# ===== ChromaDB 검색 함수들 =====
def chroma_vector_search(query, top_k=5):
    """ChromaDB Vector Search"""
    query_embedding = model.encode(query).tolist()
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['ids'][0] if results['ids'] else []


def chroma_vector_search_with_rerank(query, top_k=5, rerank_top_k=20):
    """ChromaDB Vector Search + Reranking"""
    query_embedding = model.encode(query).tolist()
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=rerank_top_k
    )
    candidate_ids = results['ids'][0] if results['ids'] else []
    
    # Reranking
    return rerank_results(query, candidate_ids, top_k)


def chroma_bm25_search(query, top_k=5):
    """ChromaDB BM25 Search"""
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [bm25_ids[i] for i in top_indices]


def chroma_bm25_search_with_rerank(query, top_k=5, rerank_top_k=20):
    """ChromaDB BM25 Search + Reranking"""
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:rerank_top_k]
    candidate_ids = [bm25_ids[i] for i in top_indices]
    
    # Reranking
    return rerank_results(query, candidate_ids, top_k)


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


def chroma_hybrid_search_with_rerank(query, top_k=5, rerank_top_k=20):
    """ChromaDB Hybrid Search + Reranking"""
    # Vector Search
    query_embedding = model.encode(query).tolist()
    vector_results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=rerank_top_k
    )
    vector_ids = vector_results['ids'][0] if vector_results['ids'] else []

    # BM25 Search
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:rerank_top_k]
    bm25_ids_top = [bm25_ids[i] for i in top_indices]

    # RRF Fusion
    rrf_scores = defaultdict(float)
    for rank, doc_id in enumerate(vector_ids):
        rrf_scores[doc_id] += 1.0 / (RRF_K + rank + 1)
    for rank, doc_id in enumerate(bm25_ids_top):
        rrf_scores[doc_id] += 1.0 / (RRF_K + rank + 1)

    # Sort by fused score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    candidate_ids = [doc_id for doc_id, _ in sorted_results[:rerank_top_k]]
    
    # Reranking
    return rerank_results(query, candidate_ids, top_k)


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
    
    # MongoDB 평가 (기존 + Rerank)
    print("\n" + "="*80)
    print("🔵 MONGODB EVALUATION")
    print("="*80)
    results.append(evaluate_metrics(queries, mongo_vector_search, "MongoDB - Vector", file_name))
    results.append(evaluate_metrics(queries, mongo_vector_search_with_rerank, "MongoDB - Vector + Rerank", file_name))
    results.append(evaluate_metrics(queries, mongo_text_search, "MongoDB - Text", file_name))
    results.append(evaluate_metrics(queries, mongo_text_search_with_rerank, "MongoDB - Text + Rerank", file_name))
    results.append(evaluate_metrics(queries, mongo_hybrid_search, "MongoDB - Hybrid", file_name))
    results.append(evaluate_metrics(queries, mongo_hybrid_search_with_rerank, "MongoDB - Hybrid + Rerank", file_name))
    
    # ChromaDB 평가 (기존 + Rerank)
    print("\n" + "="*80)
    print("🟢 CHROMADB EVALUATION")
    print("="*80)
    results.append(evaluate_metrics(queries, chroma_vector_search, "ChromaDB - Vector", file_name))
    results.append(evaluate_metrics(queries, chroma_vector_search_with_rerank, "ChromaDB - Vector + Rerank", file_name))
    results.append(evaluate_metrics(queries, chroma_bm25_search, "ChromaDB - BM25", file_name))
    results.append(evaluate_metrics(queries, chroma_bm25_search_with_rerank, "ChromaDB - BM25 + Rerank", file_name))
    results.append(evaluate_metrics(queries, chroma_hybrid_search, "ChromaDB - Hybrid", file_name))
    results.append(evaluate_metrics(queries, chroma_hybrid_search_with_rerank, "ChromaDB - Hybrid + Rerank", file_name))

    # 최종 결과 비교
    print(f"\n{'='*80}")
    print(f"📊 FINAL RESULTS COMPARISON - {file_name}")
    print(f"{'='*80}")
    print(f"{'Method':<40} {'Recall@1':<12} {'Recall@3':<12} {'Recall@5':<12} {'MRR':<10}")
    print("-" * 85)
    
    for result in results:
        print(f"{result['method']:<40} "
              f"{result['recall@1']:>10.2f}% "
              f"{result['recall@3']:>10.2f}% "
              f"{result['recall@5']:>10.2f}% "
              f"{result['mrr']:>8.4f}")
    
    print("=" * 85)
    
    # 결과 저장
    output_file = f"./evaluation_results/{file_name}_evaluation_results_with_rerank.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ Results saved to: {output_file}")