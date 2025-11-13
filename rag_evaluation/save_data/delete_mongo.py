from pymongo import MongoClient
from urllib.parse import quote_plus

# ===== 설정 =====
user_name = 'jimin_lee'
password = quote_plus("") # 
MONGO_URI = f"mongodb+srv://{user_name}:{password}@jaycluster.qvqia6n.mongodb.net/?retryWrites=true&w=majority&appName=jaycluster"

DB_NAME = "gnos_ai_base"
COLLECTION_NAME = "chunk_with_questions"

# ===== 연결 =====
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# ===== 전체 문서 삭제 =====
result = collection.delete_many({})
print(f"✅ {result.deleted_count}개 문서 삭제 완료!")
