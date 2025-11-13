# 온프레미스 RAG 평가 프레임워크 (RAG Evaluation Benchmark)

## 프로젝트 개요
 PDF를 기반으로 **고품질 RAG 테스트셋**을 자동 생성하고,  
다양한 Vector DB + 검색 전략 + Query Rewriting 기법을 체계적으로 비교한 벤치마크 코드입니다.  


- **임베딩 모델**: `Qwen/Qwen3-Embedding-0.6B`
- **비교 대상 DB**  
  - MongoDB Atlas Vector Search  
  - MongoDB Community (Aggregation $vectorSearch)  
  - ChromaDB (Persistent)  
- **검색 방식**: Vector / Keyword / Hybrid (Vector + Keyword)  
- **Query Rewriting**: Query Expansion (동의어), Query-to-Passage, Hypothetical Document Rewriting  

> **100% 재현 가능**, **단일 스크립트로 모든 조합 실행**, **결과 자동 시각화**  

## 테스트셋 생성 파이프라인

```mermaid
graph TD
    A[PDF] --> B[Docling, PDF to Markdown]
    B --> C[MarkdownHeaderTextSplitter 헤더 기준 청크 분할]
    C --> D[GPT-4o API 청크당 3개 질문 생성]
    D --> E[12480개 QA 페어 question + chunk + doc_id + page]
    E --> F[MongoDB Atlas rag_testset 컬렉션]
    E --> G[ChromaDB persistent DB]
    E --> H[MongoDB Community vector index]
