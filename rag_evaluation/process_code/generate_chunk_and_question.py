from openai import OpenAI
import json
from openai import OpenAI
import json
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language, MarkdownHeaderTextSplitter
from pathlib import Path
# ① PDF 로드



def load_and_chunk_pdf(pdf_path, chunk_size=1000, chunk_overlap=100):
    """
    PDF를 마크다운으로 변환 후 청킹하여 시각화
    
    Args:
        pdf_path (str): PDF 파일 경로
        chunk_size (int): 청크 크기
        chunk_overlap (int): 청크 오버랩
    
    Returns:
        list: 청크 리스트
    """
    # docling 으로 PDF-> 마크다운으로 변환
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    result = converter.convert(pdf_path)
    full_text = result.document.export_to_markdown(image_mode="referenced")
    
    # 헤더청킹
    # 문서를 분할할 헤더 레벨 & 이름 정의
    headers_to_split_on = [  
    (
        "#",
        "Title",
    ),  
    (
        "##",
        "Section",
    ),  
    (
        "\n\n",
        "Subsection",
    ),  
    ]
    # 헤더 spliter 정의
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    # 청크 만들기 
    chunks = markdown_splitter.split_text(full_text)
    
    # 시각화
    print(f"{'='*80}")
    print(f"총 {len(chunks)}개의 청크 생성")
    print(f"{'='*80}\n")
    
    for i, chunk in enumerate(chunks):
        print(f"[청크 {i+1}/{len(chunks)}]")
        #print(f"길이: {len(chunk)}자")
        print(f"{'-'*80}")
        print(chunk)
        print(f"{'='*80}\n")
    
    return chunks


def generate_question_for_chunk(client, content, num_questions=3, temperature=0.7):
    """청크에 대한 여러 개의 질문 생성"""
#     prompt = f"""
# You are a helpful assistant that creates multiple comprehension questions from text.

# Read the following passage and generate {num_questions} clear, diverse, and concise questions 
# that test understanding of different key ideas in it.
# Do not include answers.

# Passage:
# \"\"\"{content}\"\"\"

# Output format (JSON):
# {{"questions": ["q1", "q2", "q3"]}}
# """
    prompt = f"""
당신은 텍스트로부터 여러 개의 이해력 질문을 생성하는 도움이 되는 어시스턴트입니다.

다음 문단을 읽고 {num_questions}개의 명확하고, 다양하며, 간결한 질문을 생성하세요.
질문들은 문단의 서로 다른 핵심 아이디어에 대한 이해를 테스트해야 합니다.
답변은 포함하지 마세요.

문단:
\"\"\"{content}\"\"\"

출력 형식 (JSON):
{{"questions": ["질문1", "질문2", "질문3"]}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that writes comprehension questions for text chunks."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    try:
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        questions = result_json.get("questions", [])
    except Exception:
        # fallback: 단순 줄바꿈 기준 파싱
        raw_text = response.choices[0].message.content.strip()
        questions = [q.strip("-• ") for q in raw_text.split("\n") if len(q.strip()) > 5]

    return questions[:num_questions]

def generate_questions_for_chunks(client, chunks):
    """모든 청크에 대한 질문 생성"""
    questions = []
    
    print("질문 생성 중...\n")
    for idx, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        
        question = generate_question_for_chunk(client, content)
        
        questions.append({
            "chunk_id": idx,
            "question": question,
            "content": content
        })
        
        print(f"[{idx}] {question}")
    
    print(f"\n총 {len(questions)}개 질문 생성 완료")
    return questions


def save_questions(questions, output_path):
    """질문 데이터 저장"""
    # 디렉토리 생성
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    
    print(f"저장 완료: {output_path}")


def process_pdf_to_questions(
    file_name,
    pdf_dir,
    output_dir,
    chunk_size=1000,
    chunk_overlap=200,
    client = None
):
    """PDF를 청킹하고 질문 생성하는 전체 파이프라인"""
    
    print(f"{'='*60}")
    print(f"PDF 처리 시작: {file_name}")
    print(f"{'='*60}\n")
    
    # OpenAI 클라이언트 초기화    
    # 경로 설정
    pdf_path = f"{pdf_dir}/{file_name}.pdf"
    output_path = f"{output_dir}/{file_name}_chunk_questions.json"
    
    # 1. PDF 로드 및 청킹
    chunks = load_and_chunk_pdf(pdf_path, chunk_size, chunk_overlap)
    
    # 2. 질문 생성
    questions = generate_questions_for_chunks(client, chunks)
    
    # 3. 저장
    save_questions(questions, output_path)
    
    print(f"\n{'='*60}")
    print("처리 완료!")
    print(f"{'='*60}\n")
    
    return questions

if __name__ == "__main__":
    # 사용 예시
    file_name = '자동차보험'
    client = OpenAI(api_key = '')
    questions = process_pdf_to_questions(
        file_name=file_name,
        pdf_dir="/home/ljm/test_chunk/source_data",
        output_dir="/home/ljm/test_chunk/chunk_and_question",
        chunk_size=1000,
        chunk_overlap=100,
        client = client
    )