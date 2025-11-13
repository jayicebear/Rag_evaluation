from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from urllib.parse import quote_plus

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

# ===== 임베딩 모델 =====
model = SentenceTransformer(EMBED_MODEL)

# ===== Vector Search 함수 =====
def mongo_vector_search(query, top_k=5):
    """
    Atlas Vector Search 수행
    
    Args:
        query (str): 검색 쿼리
        top_k (int): 반환할 결과 수
    
    Returns:
        list: 검색 결과
    """
    # 1. 쿼리 임베딩 생성
    query_embedding = model.encode(query).tolist()
    
    # 2. Vector Search 파이프라인
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",  # 생성한 인덱스 이름
                "path": "chunk_embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,  # 후보 개수
                "limit": top_k
            }
        },
        {
            "$project": {
                "chunk_id": 1,
                "chunk": 1,
                "score": {"$meta": "vectorSearchScore"},  # 유사도 점수
                "_id": 0
            }
        }
    ]
    
    # 3. 검색 실행
    results = list(collection.aggregate(pipeline))
    
    return results


# ===== 테스트 =====
if __name__ == "__main__":
    # 검색 쿼리
    query = "safe sql"
    
    print(f"🔍 검색 쿼리: {query}\n")
    
    # Vector Search 실행
    results = mongo_vector_search(query, top_k=5)
    
    # 결과 출력
    print(f"✅ {len(results)}개 결과 발견:\n")
    
    for i, result in enumerate(results, 1):
        print(f"[{i}] Chunk ID: {result['chunk_id']}")
        print(f"    Score: {result['score']:.4f}")
        print(f"    Content: {result['chunk'][:150]}...")
        print()