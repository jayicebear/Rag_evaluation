from pymongo import MongoClient
from urllib.parse import quote_plus

# ===== 설정 =====
user_name = 'jimin_lee'
password = quote_plus("Superjm09!")
MONGO_URI = f"mongodb+srv://{user_name}:{password}@jaycluster.qvqia6n.mongodb.net/?retryWrites=true&w=majority&appName=jaycluster"

DB_NAME = "gnos_ai_base"
COLLECTION_NAME = "chunk_with_questions"

# ===== 연결 =====
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# ===== Text Search 함수 =====
def mongo_text_search(query, top_k=5):
    """
    Atlas Text Search 수행
    
    Args:
        query (str): 검색 쿼리
        top_k (int): 반환할 결과 수
    
    Returns:
        list: 검색 결과
    """
    pipeline = [
        {
            "$search": {
                "index": "text_search",  # 생성한 텍스트 인덱스 이름
                "text": {
                    "query": query,
                    "path": "chunk"
                }
            }
        },
        {
            "$limit": top_k
        },
        {
            "$project": {
                "chunk_id": 1,
                "chunk": 1,
                "score": {"$meta": "searchScore"},
                "_id": 0
            }
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    return results


# ===== 테스트 =====
if __name__ == "__main__":
    query = "safe sql"
    
    print(f"🔍 Text Search 쿼리: {query}\n")
    
    results = mongo_text_search(query, top_k=5)
    
    print(f"✅ {len(results)}개 결과 발견:\n")
    
    for i, result in enumerate(results, 1):
        print(f"[{i}] Chunk ID: {result['chunk_id']}")
        print(f"    Score: {result['score']:.4f}")
        print(f"    Content: {result['chunk'][:150]}...")
        print()