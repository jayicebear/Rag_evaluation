import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json

# ===== 설정 =====
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "rag_test"
CHROMA_DB_PATH = "./test_chunk/chroma_db"

# ===== ChromaDB 클라이언트 =====
client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection(name=COLLECTION_NAME)
model = SentenceTransformer(EMBED_MODEL)

# ===== 데이터 로드 =====
file_name = 'safe_sql'
json_file = f"./chunk_and_question/{file_name}_chunk_questions.json"

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 80)
print("🔍 상세 디버깅 - 첫 3개 질문")
print("=" * 80)

# 처음 3개 청크의 첫 번째 질문으로 테스트
for i in range(min(3, len(data))):
    item = data[i]
    local_chunk_id = item['chunk_id']
    full_chunk_id = f"{file_name}_{local_chunk_id}"
    
    # 실제 content 확인
    actual_doc = collection.get(ids=[full_chunk_id], include=["documents"])
    
    print(f"\n{'='*80}")
    print(f"[테스트 {i+1}] chunk_id: {full_chunk_id}")
    print(f"{'='*80}")
    
    print(f"\n📄 실제 Content (처음 200자):")
    print(actual_doc['documents'][0][:200] if actual_doc['documents'] else "❌ 문서 없음")
    
    # 첫 번째 질문
    question = item['question'][0]
    print(f"\n❓ 질문:")
    print(f"   {question}")
    
    # Vector Search
    query_embedding = model.encode(question).tolist()
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "distances"]
    )
    
    print(f"\n📊 Vector Search Top-5 결과:")
    for rank, (result_id, distance, doc) in enumerate(zip(
        vector_results['ids'][0], 
        vector_results['distances'][0],
        vector_results['documents'][0]
    ), 1):
        is_correct = "✅" if result_id == full_chunk_id else "❌"
        print(f"  {is_correct} [{rank}] {result_id} (distance: {distance:.4f})")
        print(f"       {doc[:100]}...")
    
    # BM25
    from rank_bm25 import BM25Okapi
    import numpy as np
    
    all_docs = collection.get(include=["documents"])
    documents = all_docs['documents']
    ids = all_docs['ids']
    tokenized_corpus = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = question.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:5]
    
    print(f"\n📊 BM25 Top-5 결과:")
    for rank, idx in enumerate(top_indices, 1):
        result_id = ids[idx]
        score = scores[idx]
        is_correct = "✅" if result_id == full_chunk_id else "❌"
        print(f"  {is_correct} [{rank}] {result_id} (score: {score:.4f})")
        print(f"       {documents[idx][:100]}...")
    
    # 정답 순위 확인
    try:
        vector_rank = vector_results['ids'][0].index(full_chunk_id) + 1
        print(f"\n🎯 Vector Search: 정답이 {vector_rank}위")
    except ValueError:
        print(f"\n❌ Vector Search: 정답이 Top-5 안에 없음")
    
    try:
        bm25_rank = [ids[i] for i in top_indices].index(full_chunk_id) + 1
        print(f"🎯 BM25 Search: 정답이 {bm25_rank}위")
    except ValueError:
        print(f"❌ BM25 Search: 정답이 Top-5 안에 없음")

print("\n" + "=" * 80)