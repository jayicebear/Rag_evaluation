import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
import os

# ===== 설정 =====
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "rag_test"
CHROMA_DB_PATH = "./chroma/chroma_db"

# ===== 저장 디렉토리 생성 =====
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# ===== ChromaDB 클라이언트 생성 =====
client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

print(f"📂 ChromaDB 저장 경로: {os.path.abspath(CHROMA_DB_PATH)}")

# ===== 임베딩 모델 =====
model = SentenceTransformer(EMBED_MODEL)

# ===== 컬렉션 생성 (기존 것 있으면 삭제) =====
try:
    client.delete_collection(name=COLLECTION_NAME)
    print("✅ 기존 컬렉션 삭제 완료")
except:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# ===== 여러 문서 저장 =====
file_names = ['safe_sql', '성희롱_예방', 'raptor','short_story','국방부_업무보고','자동차보험']  # ✅ 모든 문서 나열

for file_name in file_names:
    input_file = f"./chunk_and_question/{file_name}_chunk_questions.json"
    
    # 파일이 없으면 스킵
    if not os.path.exists(input_file):
        print(f"⚠️ {input_file} 파일 없음, 스킵")
        continue
    
    print(f"\n📂 처리 중: {file_name}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✅ 총 {len(data)}개 청크 로드 완료")
    
    # ===== ChromaDB에 삽입 =====
    documents = []
    embeddings = []
    ids = []
    metadatas = []
    
    for item in data:
        chunk_text = item["content"]
        chunk_id = item["chunk_id"]
        
        # 임베딩 생성
        embedding = model.encode(chunk_text).tolist()
        
        # 복합 ID 생성
        full_chunk_id = f"{file_name}_{chunk_id}"
        
        documents.append(chunk_text)
        embeddings.append(embedding)
        ids.append(full_chunk_id)
        metadatas.append({
            "document_name": file_name,
            "local_chunk_id": chunk_id
        })
        
        if len(documents) % 10 == 0:
            print(f"  처리 중... {len(documents)}/{len(data)}")
    
    # 배치로 삽입
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"✅ {len(documents)}개 문서 삽입 완료!")

print(f"\n" + "="*80)
print(f"📊 최종 컬렉션 문서 수: {collection.count()}")
print(f"💾 데이터 저장 위치: {os.path.abspath(CHROMA_DB_PATH)}")

# 저장된 문서별 개수 확인
all_data = collection.get(include=["metadatas"])
doc_counts = {}
for meta in all_data['metadatas']:
    doc_name = meta['document_name']
    doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1

print(f"\n📊 문서별 청크 수:")
for doc_name, count in doc_counts.items():
    print(f"  - {doc_name}: {count}개")
print("="*80)