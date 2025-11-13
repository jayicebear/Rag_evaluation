
import os
import json
import argparse
from tqdm import tqdm
import openai

# ===== OpenAI API 설정 =====
openai.api_key = ''  # 여기에 API 키 입력


# ===== argparse 설정 =====
parser = argparse.ArgumentParser(description="Query rewriting methods")
parser.add_argument("--file_name", type=str, required=True, help="파일 이름")
parser.add_argument("--mode", type=str, choices=["rewrite", "expansion","passage"], default="rewrite",
                    help="rewrite 모드 선택")
args = parser.parse_args()

file_name = args.file_name
mode = args.mode

# ===== 파일 경로 설정 =====
input_path = f"./chunk_and_question/{file_name}_chunk_questions.json"
output_path = f"./rewritten_query/{file_name}_{mode}_rewritten.json"

# ===== GPT 모델 설정 =====
MODEL_NAME = "gpt-4o-mini"

# ===== Query Rewriting 함수 =====
def rewrite_query(question: str):
    prompt = f"""
You are an expert in question rewriting for information retrieval.
Take the following question and rewrite it into a single, more elaborate, contextually rich version. 
Keep the same meaning, but make it sound more natural, specific, or insightful, by adding brief context or clarifying what is being asked.

Question: "{question}"

Return only the rewritten question as plain text (no JSON, no quotes).
"""
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# ===== Query Expansion 함수 =====
def query_expansion(question: str):
    prompt = f"""
You are an expert in query expansion for information retrieval.
Take the following question and expand it into a single, more detailed query 
that would improve retrieval performance. 
Preserve the original meaning, but enrich it by including related terms, 
synonyms, or relevant contextual phrases that could help a search more comprehensive results.

Question: "{question}"

Return only the expanded query as plain text (no JSON, no quotes).
"""
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def query_to_passage(question: str):
    """
    검색용 질문(query)을 기반으로 관련 내용을 설명하는 짧은 passage를 생성하는 함수.
    즉, '이 질문이 다루는 주제나 핵심 개념'을 문맥적으로 기술하는 문단을 생성함.
    GPT-4o-mini 사용.
    """
    prompt = f"""
You are an expert in question rewriting for information retrieval.
Take the following search query and generate a short, coherent passage (2–3 sentences)
that could provide context or background information relevant to answering the query.

The passage should not directly answer the question,
but should sound like part of a document or article where the answer could be found.
Do not include the question itself — write only the passage text.

Query: "{question}"

Return only the passage text (no JSON, no quotes).
"""

    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8  # 다양성을 조금 높임
    )

    passage = response.choices[0].message.content.strip()
    return passage

# ===== JSON 로드 =====
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"{len(data)}개 청크 로드 완료")

# ===== Query Rewriting 수행 =====
rewritten_data = []

for item in tqdm(data, desc=f"Processing ({mode})"):
    chunk_id = item["chunk_id"]
    content = item["content"]
    original_questions = item["question"]

    rewritten_questions = []
    for q in original_questions:
        if mode == "rewrite":
            new_q = rewrite_query(q)
        elif mode == "passage":
            new_q = query_to_passage(q)
        else:
            new_q = query_expansion(q)
        rewritten_questions.append(new_q)

    rewritten_data.append({
        "chunk_id": chunk_id,
        "content": content,
        "question": rewritten_questions
    })

# ===== 결과 저장 =====
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(rewritten_data, f, ensure_ascii=False, indent=4)

print(f"결과 저장 완료: {output_path}")
