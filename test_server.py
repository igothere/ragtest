#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "서버가 정상 작동 중입니다!"}), 200

@app.route('/upload', methods=['POST'])
def upload_test():
    print("📥 업로드 요청 수신됨")
    
    if 'file' not in request.files:
        print("❌ 파일이 없습니다")
        return jsonify({"error": "파일이 없습니다"}), 400
    
    file = request.files['file']
    print(f"📄 파일명: {file.filename}")
    
    if file.filename == '':
        print("❌ 파일이 선택되지 않았습니다")
        return jsonify({"error": "파일이 선택되지 않았습니다"}), 400
    
    print("✅ 파일 업로드 테스트 성공!")
    return jsonify({"message": "파일 업로드 테스트 성공", "filename": file.filename}), 200

if __name__ == '__main__':
    print("🚀 테스트 서버 시작...")
    app.run(host='0.0.0.0', port=5001, debug=True)