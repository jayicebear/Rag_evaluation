# 온프레미스 RAG 평가 프레임워크 (RAG Evaluation Benchmark)

## 프로젝트 개요
 PDF를 기반으로 **RAG 테스트셋**을 자동 생성하고,  
Chroma DB 와 Mongo DB 의 RAG 정확도 비교한 벤치마크 코드입니다.(+Query Rewriting 기법)  


- **임베딩 모델**: `Qwen/Qwen3-Embedding-0.6B`
- **비교 대상 DB**  
  - MongoDB Atlas Search  
  - MongoDB Community Search
  - ChromaDB  
- **검색 방식**: Vector / Keyword / Hybrid (Vector + Keyword)
- **비교 방식**:<br>
  `Vector`(chroma db embedding vs mongo DB vector search)<br>
  `Keyword`(chroma db bm25 vs mongo db keyword search<br>
  `hybrid`(chroma db vector + bm25 score search vs mongo db hybrid rank fusion search)<br>
- **Query Rewriting**: Query Expansion, Query-to-Passage, Hypothetical Document Rewriting  

> **100% 재현 가능**, **단일 스크립트로 모든 조합 실행**, **결과 자동 시각화**  

## RAG 바교 테스트 파이프라인

```mermaid
graph TD
    A[PDF] --> B[Docling PDF->Markdown]
    B --> C[MarkdownHeaderTextSplitter 헤더 기준 청크 분할]
    C --> D[GPT API 청크당 질문 생성]
    D --> E[12480개 QA 페어 question + chunk + doc_id + page]
    E --> F[MongoDB Atlas]
    E --> G[ChromaDB]
    E --> H[MongoDB Community]
