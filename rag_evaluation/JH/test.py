"""
VectorSearchService 완전 테스트
"""
import asyncio
import os

from beanie import init_beanie
from dotenv import load_dotenv
from pymongo.asynchronous.mongo_client import AsyncMongoClient

from app.models.document import User, Conversation, Document
from app.models.document_chunk import DocumentChunk
from app.services.vector_search_service import VectorSearchService

load_dotenv()



async def test_1_mongodb_connection(client):
    """1단계: MongoDB 연결 및 데이터 확인"""
    print("=" * 70)
    print("  TEST 1: MongoDB 연결 및 데이터")
    print("=" * 70)

    try:
        db = client[os.getenv("MONGODB_DATABASE")]

        chunks_count = await db.document_chunks.count_documents({})
        print(f"✅ document_chunks 컬렉션 접근 성공")
        print(f"   - 전체 청크 수: {chunks_count}개")

        if chunks_count == 0:
            print(f"   ⚠️ 청크 데이터가 없습니다!")
            print(f"   → test_insert_dummy_chunks.py를 실행하세요")
            return False

        sample = await db.document_chunks.find_one({})
        print(f"\n✅ 샘플 청크 확인:")
        print(f"   - _id: {sample['_id']}")
        print(f"   - documentId: {sample.get('documentId')}")
        print(f"   - content: {sample.get('content', '')[:50]}...")
        print(f"   - embedding: {len(sample.get('embedding', []))}차원")

        docs_count = await db.documents.count_documents({})
        print(f"\n✅ documents 컬렉션 접근 성공")
        print(f"   - 전체 문서 수: {docs_count}개")

        print("\n✅ TEST 2 통과\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 2 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False



async def test_2_vector_search():
    """2단계: 실제 벡터 검색 테스트"""
    print("=" * 70)
    print("  TEST 2: 벡터 검색 실행")
    print("=" * 70)

    try:
        vector_search_service = VectorSearchService()

        test_queries = [
            ("AI 윤리", "68ecb905d547dd8b981f2944"),
            ("FastAPI 개발", "68ecb905d547dd8b981f2944"),
            ("MongoDB 벡터", "68ecb905d547dd8b981f2944"),
            ("데이터셋", "68ecb905d547dd8b981f2944"),
        ]

        success_count = 0

        for query, user_id in test_queries:
            print(f"\n🔍 검색 쿼리: '{query}'")
            print(f"   사용자 ID: {user_id}")

            try:
                results = await vector_search_service.search_documents(
                    user_id=user_id,
                    query=query,
                    limit=5
                )

                documents = results.get('documents', [])
                print(f"   ✅ 검색 성공: {len(documents)}개 문서 발견")

                if documents:
                    success_count += 1
                    for i, doc in enumerate(documents[:3], 1):
                        print(f"\n      [{i}] {doc['originFilename']}")
                        print(f"          relevance: {doc['relevance']:.4f}")
                        print(f"          evidences: {doc['evidenceCount']}개")
                        # print("-----------------evidences-----------------------")
                        # print(doc.get('evidences'))
                        # print("-----------------evidences-----------------------")
                        if doc.get('evidences'):
                            evidence = doc['evidences'][0]
                            # print(evidence)
                            content_preview = evidence.get('content', '')[:80]
                            print(f"          내용: {content_preview}...")
                else:
                    print(f"   ⚠️ 검색 결과 없음")

            except Exception as e:
                print(f"   ❌ 검색 실패: {type(e).__name__} - {e}")

        if success_count > 0:
            print(f"\n✅ TEST 4 통과: {success_count}/{len(test_queries)}개 쿼리 성공\n")
            return True
        else:
            print(f"\n⚠️ TEST 4: 모든 쿼리에서 결과 없음\n")
            return False

    except Exception as e:
        print(f"\n❌ TEST 4 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


async def init_beanie_for_test():
    """Beanie 초기화"""
    mongo_url = os.getenv("MONGODB_URL")
    db_name = os.getenv("MONGODB_DATABASE")

    print(f"🔌 MongoDB 연결 중... ({db_name})\n")

    client = AsyncMongoClient(mongo_url)
    database = client[db_name]

    await init_beanie(
        database=database,
        document_models=[User, Conversation, Document, DocumentChunk]
    )

    print("✅ Beanie 초기화 완료\n")
    return client


async def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "=" * 70)
    print("  VectorSearchService 통합 테스트")
    print("=" * 70)
    print()

    client = None
    results = []

    try:
        client = await init_beanie_for_test()

        results.append(("MongoDB 연결", await test_1_mongodb_connection(client)))
        results.append(("벡터 검색", await test_2_vector_search()))

    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류: {e}\n")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            await client.close()

    print("\n" + "=" * 70)
    print("  테스트 결과 요약")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"  {status}  {name}")

    print()
    print(f"  총 {passed}/{total}개 테스트 통과")
    print("=" * 70)

    if passed == total:
        print("\n🎉 모든 테스트 통과!\n")
    else:
        print(f"\n⚠️ {total - passed}개 테스트 실패\n")


if __name__ == "__main__":
    asyncio.run(run_all_tests())



