
# ===== 설정 =====
#MONGO_URI = "mongodb://zium:zium1207%21%21@192.168.0.43:27017/admin"
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import json
import os

# ===== 설정 =====
# 43 번 서버
#MONGO_URI = f"mongodb://zium:zium1207!!@192.168.0.43:27017/?authSource=admin"
MONGO_URI = f"mongodb://zium:zium1207%21%21@192.168.0.43:27017/admin"
DB_NAME = "rag_test"
COLLECTION_NAME = "chunks_for_test"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DATA_DIR = "./chunk_and_question"  # JSON 폴더 경로

# ===== 연결 =====
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# ===== 임베딩 모델 =====
model = SentenceTransformer(EMBED_MODEL)

# ===== 폴더 내 JSON 파일 탐색 =====
json_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
print(f"📂 감지된 JSON 파일 {len(json_files)}개:")
for f in json_files:
    print(f"  - {f}")

# ===== 모든 파일 반복 삽입 =====
for file_name in json_files:
    # 파일 이름(확장자 제외)
    base_name = os.path.splitext(file_name)[0]
    input_path = os.path.join(DATA_DIR, file_name)

    print(f"\n🔹 파일 로드 중: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✅ 총 {len(data)}개 청크 로드 완료")

    # 기존 동일 prefix 데이터 삭제 (선택 사항)
    collection.delete_many({"chunk_id": {"$regex": f"^{base_name}_"}})
    print(f"🗑️ 기존 {base_name}_* 문서 삭제 완료")

    # 문서 변환 및 임베딩
    docs_to_insert = []
    for item in data:
        chunk_text = item["content"]
        questions = item.get("question", [])
        chunk_id = f"{base_name}_{item['chunk_id']}"  # 파일명_prefix

        embedding = model.encode(chunk_text).tolist()

        docs_to_insert.append({
            "chunk_id": chunk_id,
            "chunk": chunk_text,
            "chunk_embedding": embedding
        })

    # MongoDB 삽입
    if docs_to_insert:
        collection.insert_many(docs_to_insert)
        print(f"✅ {len(docs_to_insert)}개 문서 MongoDB에 삽입 완료!")

# ===== 샘플 확인 =====
print("\n📘 샘플 확인:")
for doc in collection.find().limit(3):
    print(f"- {doc['chunk_id']}: {doc['chunk'][:100]}...")
