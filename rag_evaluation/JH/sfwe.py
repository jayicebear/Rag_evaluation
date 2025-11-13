"""
MongoDB 하이브리드 검색 서비스 (self-managed mongod + mongot)
Vector Search + Full-Text Search with Reciprocal Rank Fusion (RRF)
"""
import logging
from collections import defaultdict
from functools import lru_cache
from typing import List, Dict, Any, Optional

from bson import ObjectId

from app.models.document_chunk import DocumentChunk
from app.services.embedding_service import get_embedding_service_singleton
from app.services.reranker_service import RerankerService

logger = logging.getLogger(__name__)

RRF_K = 60


def _theoretical_rrf_max(num_pipelines: int, weights: Optional[Dict[str, float]] = None) -> float:
    """각 파이프라인이 모두 1위를 차지한다고 가정했을 때의 RRF 이론적 최대치"""
    if not weights:
        return num_pipelines * (1.0 / (RRF_K + 1))
    return sum(float(w) * (1.0 / (RRF_K + 1)) for w in weights.values())


def _normalize_rrf(rrf_score: float, num_pipelines: int, weights: Optional[Dict[str, float]] = None) -> float:
    """UI 표시용 정규화 점수(0~1)"""
    denom = _theoretical_rrf_max(num_pipelines, weights)
    if denom <= 0:
        return 0.0
    v = rrf_score / denom
    return 1.0 if v > 1.0 else v


class VectorSearchService:
    """MongoDB Hybrid Search (Vector + Full-Text with RRF)"""

    def __init__(self):
        self.embedding_service = get_embedding_service_singleton()
        logger.info("🔍 VectorSearchService (MongoDB Hybrid Search) initialized")

    async def search_documents(
            self,
            user_id: str,
            query: str,
            limit: int = 3,
            reranker_service: Optional[RerankerService] = None,
            **kwargs  # 호환성 유지(사용 안 함)
    ) -> Dict[str, Any]:
        """
        MongoDB 하이브리드 검색 (벡터 + 텍스트 + RRF)

        Returns:
            {"documents": [ ... ]}  # 문서별 그룹핑 결과
        """
        try:
            logger.info(f"🔍 Hybrid search (Vector+Text) - User: {user_id}, Query: {query[:50]}...")

            # 1) 쿼리 임베딩
            query_embedding = await self.embedding_service.encode_single(query)
            logger.info(f"✅ Query embedding generated (dim={len(query_embedding)})")

            # 2) MongoDB Hybrid Search (RRF)
            search_results = await self._mongo_hybrid_search(
                user_id=user_id,
                query=query,
                query_embedding=query_embedding,
                limit=limit
            )

            if not search_results:
                logger.info("📚 No results found")
                return {"documents": []}

            # 3) 문서별 그룹핑
            grouped_documents = self._group_chunks_by_document(search_results)
            logger.info(f"📂 Grouped into {len(grouped_documents)} documents")

            # 4) 리랭킹(옵션)
            if reranker_service and grouped_documents:
                logger.info("🔄 Applying reranking...")
                grouped_documents = await self._apply_reranking(
                    reranker_service, query, grouped_documents, limit
                )

            # 5) 최종 결과
            final_results = grouped_documents[:limit]

            logger.info(f"✅ Search completed - Found {len(final_results)} documents")
            for i, doc in enumerate(final_results):
                logger.info(
                    f"  📄 [{i + 1}] {doc['originFilename']} "
                    f"(relevance: {doc['relevance']:.6f}, evidences: {doc['evidenceCount']})"
                )
                for j, ev in enumerate(doc['evidences'][:2]):
                    logger.info(
                        f"    📝 Evidence {j + 1}: page={ev['page']}, rrf={ev['relevanceScore']:.6f}"
                    )
                    logger.info(f"       Content: {ev['content'][:100]}...")

            return {"documents": final_results}

        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            raise

    async def _mongo_hybrid_search(
            self,
            user_id: str,
            query: str,
            query_embedding: List[float],
            limit: int
    ) -> List[Dict]:
        """MongoDB Hybrid Search 실행 (Vector + Text with RRF)"""
        try:
            user_oid = ObjectId(user_id)
        except Exception:
            logger.error(f"Invalid user_id: {user_id}")
            return []

        # 입력 파이프라인(Selection + Ranked) 내부에 필터/리밋 포함
        per_stage_limit = max(limit * 2, 40)
        num_candidates = max(limit * 10, 200)

        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vector": [
                                {
                                    "$vectorSearch": {
                                        "index": "vector_idx_qwen3",
                                        "path": "embedding",
                                        "queryVector": query_embedding,
                                        "numCandidates": num_candidates,
                                        "limit": per_stage_limit
                                    }
                                },
                                {
                                    "$match": {
                                        "$or": [{"owner": user_oid}, {"isGlobal": True}]
                                    }
                                },
                                {"$limit": per_stage_limit}
                            ],
                            "text": [
                                {
                                    "$search": {
                                        "index": "search_idx",
                                        "text": {"query": query, "path": "content"}
                                    }
                                },
                                {
                                    "$match": {
                                        "$or": [{"owner": user_oid}, {"isGlobal": True}]
                                    }
                                },
                                {"$limit": per_stage_limit}
                            ]
                        }
                    },
                    "scoreDetails": True  # scoreDetails.value 사용 가능
                }
            },
            # ✅ RRF 점수: { $meta: "score" } / 상세: { $meta: "scoreDetails" }
            {
                "$addFields": {
                    "rrf": {"$meta": "scoreDetails"},
                    "rrfScore": {"$meta": "score"}
                }
            },

            {
                "$match": {
                    "rrfScore": {"$gte": 0.015}  # 최소 품질 기준
                }
            },
            # 이후 조인
            {
                "$lookup": {
                    "from": "documents",
                    "localField": "documentId",
                    "foreignField": "_id",
                    "as": "document"
                }
            },
            {"$unwind": "$document"},
            {"$sort": {"rrfScore": -1}}
        ]

        logger.info("🔍 Executing MongoDB $rankFusion (Vector + Text with RRF)...")
        results = await DocumentChunk.aggregate(pipeline).to_list()
        logger.info(f"📚 Found {len(results)} chunks (after $rankFusion RRF ranking)")

        if results:
            logger.info("=" * 80)
            logger.info("🔍 MongoDB Pipeline Results (raw):")
            for i, result in enumerate(results[:3]):
                logger.info(f"\n[Chunk {i+1}]")
                logger.info(f"  _id: {result.get('_id')}")
                logger.info(f"  content: {result.get('content', '')[:100]}...")
                logger.info(f"  rrfScore: {result.get('rrfScore')}")
                logger.info(f"  rrf (scoreDetails): {result.get('rrf')}")
                logger.info(f"  page: {result.get('page')}")
                logger.info(f"  document: {result.get('document', {}).get('originFilename')}")
            logger.info("=" * 80)

        return results

    def _group_chunks_by_document(self, search_results: List[Dict]) -> List[Dict]:
        """청크를 문서별로 그룹핑"""
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for item in search_results:
            doc_id = str(item["document"]["_id"])
            grouped[doc_id].append(item)

        # 입력 파이프라인 수/가중치 (vector, text 두 개 / 동일 가중치)
        num_pipelines = 2
        weights = None  # {"vector": 1.0, "text": 1.0} 형태 관리 시 전달

        results: List[Dict[str, Any]] = []
        for doc_id, items in grouped.items():
            doc = items[0]["document"]

            evidences = []
            max_rrf = 0.0

            for item in items:
                rrf_meta = item.get("rrf", {})
                rrf_value = rrf_meta.get("value") if isinstance(rrf_meta, dict) else None
                rrf = float(rrf_value) if rrf_value is not None else float(item.get("rrfScore", 0.0))

                if rrf > 0.0:
                    evidences.append({
                        "chunkId": str(item["_id"]),
                        "page": int(item.get("page", 0)),
                        "content": item.get("content", ""),
                        "relevanceScore": rrf,  # 원 RRF 점수(≈0.03대)
                        "uiScore": _normalize_rrf(rrf, num_pipelines, weights)  # 0~1 정규화
                    })
                    if rrf > max_rrf:
                        max_rrf = rrf

            if evidences:
                evidences.sort(key=lambda x: x["relevanceScore"], reverse=True)
                results.append({
                    "id": doc_id,
                    "originFilename": doc.get("originFilename", ""),
                    "filePath": doc.get("filePath", ""),
                    "owner": str(doc.get("owner", "")),
                    "isGlobal": doc.get("isGlobal", False),
                    "status": doc.get("status", ""),
                    "summary": doc.get("summary"),
                    "chunkingType": doc.get("chunkingType"),
                    "createdAt": doc.get("createdAt").isoformat() if doc.get("createdAt") else None,
                    "relevance": _normalize_rrf(max_rrf, num_pipelines, weights),  # 문서 대표(0~1)
                    "maxRelevanceScore": max_rrf,  # 원 RRF 점수(로그/디버그용)
                    "evidenceCount": len(evidences),
                    "evidences": evidences
                })

        # 문서 정렬은 원 RRF 점수 기준 유지
        results.sort(key=lambda x: x["maxRelevanceScore"], reverse=True)
        return results

    async def _apply_reranking(
            self,
            reranker_service: RerankerService,
            query: str,
            documents: List[Dict],
            top_k: int
    ) -> List[Dict]:
        """리랭킹 적용"""
        try:
            docs_for_rerank = []
            for doc in documents:
                representative_text = (doc.get("summary") or "") + " "
                if doc.get("evidences"):
                    representative_text += " ".join(e["content"] for e in doc["evidences"])
                docs_for_rerank.append({**doc, "content": representative_text})

            reranked_docs = await reranker_service.rerank(query, docs_for_rerank, top_k)

            cleaned_docs = []
            for doc in reranked_docs:
                doc.pop("content", None)
                if "rerank_score" in doc:
                    doc["relevance"] = doc["rerank_score"]
                    logger.info(
                        f"  📄 {doc['originFilename']}: "
                        f"RRF={doc['maxRelevanceScore']:.6f} → rerank={doc['relevance']:.6f}"
                    )
                    doc.pop("rerank_score", None)
                cleaned_docs.append(doc)

            logger.info(f"✅ Reranking completed - {len(cleaned_docs)} documents reordered")
            return cleaned_docs

        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return documents[:top_k]


@lru_cache(maxsize=1)
def get_vector_search_service_singleton() -> VectorSearchService:
    """싱글톤 인스턴스"""
    logger.info("🏭 Creating VectorSearchService singleton")
    return VectorSearchService()