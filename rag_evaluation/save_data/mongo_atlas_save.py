from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import json
from urllib.parse import quote_plus

# ===== 설정 =====
user_name = 'jimin_lee'
password = quote_plus("")
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

# ===== JSON 데이터 로드 =====
file_name = '자동차보험'
input_file = f"./chunk_and_question/{file_name}_chunk_questions.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ 총 {len(data)}개 청크 로드 완료\n")

# ===== MongoDB에 삽입 =====
docs_to_insert = []

for item in data:
    chunk_text = item["content"]
    questions = item.get("question", [])
    original_chunk_id = item["chunk_id"]
    # content 임베딩 생성
    embedding = model.encode(chunk_text).tolist()

    docs_to_insert.append({
        "chunk_id": f"{file_name}_{original_chunk_id}",
        "chunk": chunk_text,
        "chunk_embedding": embedding
    })

# MongoDB 삽입
collection.insert_many(docs_to_insert)
print(f"✅ {len(docs_to_insert)}개 문서 MongoDB에 삽입 완료!")

# ===== 확인 =====
for doc in collection.find().limit(2):
    print("\n--- 저장된 문서 예시 ---")
    print(f"chunk_id: {doc['chunk_id']}")
    print(f"chunk(요약): {doc['chunk'][:120]}...")