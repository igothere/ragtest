# api_server.py

import os
import subprocess
import uuid # uuid 임포트
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests # API 호출을 위해 추가
import psycopg2
from sentence_transformers import SentenceTransformer
from rag import normalize_text 
# from werkzeug.utils import secure_filename

# 설정
UPLOAD_FOLDER = "./docs"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md'}
PROCESSING_SCRIPT = 'rag.py'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "파일이 없습니다"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "파일이 선택되지 않았습니다"}), 400
        
    if file and allowed_file(file.filename):
        # original_filename = secure_filename(file.filename)
        original_filename = file.filename
        
        # 고유 파일명 생성 (uuid + 확장자)
        file_ext = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_ext}"
        
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            # 1. 파일을 고유한 이름으로 디렉토리에 저장
            file.save(save_path)
            print(f"✅ 파일 저장 성공: {save_path} (원본: {original_filename})")

            # 2. 환경변수 설정
            env = os.environ.copy()
            env['OLLAMA_ENDPOINT'] = "https://api.hamonize.com/ollama/api/chat"
            env['OLLAMA_MODEL'] = "airun-chat:latest"
            env['USE_SUMMARIZATION_CHUNKING'] = "true"

            # 3. rag.py 스크립트에 고유 파일명과 원본 파일명을 모두 전달
            print(f"🚀 {PROCESSING_SCRIPT} 실행하여 {unique_filename} 처리 시작...")
            
            # 가상환경의 python 경로 사용
            python_path = os.path.join(os.getcwd(), 'venv', 'bin', 'python')
            if not os.path.exists(python_path):
                python_path = 'python'  # 가상환경이 없으면 기본 python 사용
            
            # 표 처리 RAG 스크립트 사용
            table_script = 'rag_with_tables.py'
            if os.path.exists(table_script):
                script_to_use = table_script
                print(f"  📊 표 처리 RAG 스크립트 사용: {script_to_use}")
            else:
                script_to_use = PROCESSING_SCRIPT
                print(f"  📝 기본 RAG 스크립트 사용: {script_to_use}")
            
            result = subprocess.run(
                [python_path, script_to_use, unique_filename, original_filename], # 인자 2개 전달
                capture_output=True,
                text=True,
                check=True,
                env=env,  # 환경변수 전달
                cwd=os.getcwd()  # 현재 디렉토리에서 실행
            )
            
            print(f"📜 스크립트 출력:\n{result.stdout}")
            return jsonify({"message": f"'{original_filename}' 파일 처리 성공", "output": result.stdout}), 200

        except subprocess.CalledProcessError as e:
            print(f"❌ 스크립트 실행 오류:")
            print(f"   반환 코드: {e.returncode}")
            print(f"   표준 출력: {e.stdout}")
            print(f"   표준 에러: {e.stderr}")
            return jsonify({
                "error": "파일 처리 중 오류 발생", 
                "details": e.stderr,
                "stdout": e.stdout,
                "returncode": e.returncode
            }), 500
        except Exception as e:
            print(f"❌ 서버 오류: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
            
    return jsonify({"error": "허용되지 않는 파일 형식입니다"}), 400
  
  # --- 서버 시작 시 모델과 DB 설정을 미리 준비 ---
print("🤖 RAG 검색용 임베딩 모델 로딩 중...")
try:
    model = SentenceTransformer("nlpai-lab/KURE-v1")
    print("✅ 검색용 모델 로딩 완료.")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    model = None

DB_CONFIG = {
    "host": "localhost", "port": "5432", "dbname": "ragtest",
    "user": "eden", "password": "qwer123"
}

# --- Ollama 설정 추가 ---
OLLAMA_ENDPOINT = "https://api.hamonize.com/ollama/api/chat"
OLLAMA_MODEL = "airun-chat:latest" # 사용하려는 Ollama 모델명
# -------------------------

# 기존 ask_question 또는 chat 함수를 아래 내용으로 교체합니다.
@app.route('/chat', methods=['POST'])
def chat_with_doc():
    if not model:
        return jsonify({"error": "모델이 로드되지 않았습니다."}), 500

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "질문이 없습니다."}), 400

    original_question = data['question']
    print(f"🔍 수신된 질문: {original_question}")
    
    # 질문도 문서와 동일하게 정규화합니다.
    normalized_question = normalize_text(original_question)
    print(f"🔍 정규화된 질문: {normalized_question}")

    try:
        # 1. 질문 벡터화 및 유사 문서 검색
        print("  - 1. 질문 벡터화 및 유사 문서 검색 중...")
        question_embedding = model.encode(normalized_question).tolist()
        embedding_str = "[" + ",".join(map(str, question_embedding)) + "]"

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        # id, 원본 파일명, 청크 번호, 유사도도 함께 가져오도록 SQL 수정
        sql = """
            SELECT 
                id, 
                content, 
                original_filename, 
                chunk_index,
                1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT 5;
        """
        cur.execute(sql, (embedding_str, embedding_str))
        results = cur.fetchall()
        cur.close()
        conn.close()

        # 검색 결과를 나중에 반환하기 위해 저장
        sources = [
            {
                "id": row[0],
                "content": row[1],
                "filename": row[2],
                "chunk": row[3],
                "similarity": f"{row[4]:.4f}"
            }
            for row in results
        ]

        if not sources:
            return jsonify({"answer": "관련된 정보를 문서에서 찾을 수 없습니다.", "sources": []})

        # 2. 프롬프트 생성
        print("  - 2. AI 프롬프트 생성 중...")
        context = "\n\n".join([source['content'] for source in sources])
        
        prompt = f"""
        You are an AI assistant that provides accurate and reliable answers to user questions, primarily by referring to the given 'document content.'
You should actively use the 'document content' when answering, but you may also utilize external knowledge or web information if necessary.
All answers must be written in Korean.

        --- 문서 내용 ---
        {context}
        -----------------

        --- 질문 ---
        {normalized_question}
        ------------

        답변:
        """

        # 3. Ollama API 호출
        print(f"  - 3. Ollama API ({OLLAMA_MODEL}) 호출 중...")
        ollama_payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        
        response = requests.post(OLLAMA_ENDPOINT, json=ollama_payload, timeout=120)
        response.raise_for_status()
        
        response_data = response.json()
        answer = response_data['message']['content']
        
        print("  - 4. AI 답변 수신 완료.")
        
        # 4. 최종 답변과 근거 문서를 함께 반환
        return jsonify({"answer": answer, "sources": sources})

    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama API 호출 오류: {e}")
        return jsonify({"error": "AI 모델 서버에 연결할 수 없습니다."}), 500
    except Exception as e:
        print(f"❌ 질문 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "답변 생성 중 서버에서 오류가 발생했습니다."}), 500


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=5001, debug=True)