// src/app/ask/page.tsx
"use client";

import { useState, FormEvent } from "react";
import axios from "axios";

// 근거 문서(소스)의 타입을 정의
interface Source {
  id: number;
  content: string;
  filename: string;
  chunk: number;
  similarity: string;
}

export default function AskPage() {
  const [question, setQuestion] = useState<string>("");
  const [answer, setAnswer] = useState<string>("");
  const [sources, setSources] = useState<Source[]>([]); // 근거 문서 목록 상태
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!question.trim()) {
      setError("질문을 입력해주세요.");
      return;
    }

    setIsLoading(true);
    setAnswer("");
    setSources([]); // 이전 결과 초기화
    setError("");

    try {
      const response = await axios.post("http://localhost:5001/chat", { question });
      setAnswer(response.data.answer); // AI 답변 저장
      setSources(response.data.sources); // 근거 문서 목록 저장
    } catch (err: any) {
      const errorMsg = err.response?.data?.error || "답변을 가져오는 중 오류가 발생했습니다.";
      setError(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter 키가 눌렸고, Shift 키는 눌리지 않았을 때
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // 기본 동작(줄바꿈) 방지

      // 로딩 중이 아닐 때만 form의 submit 이벤트를 강제로 발생시킴
      if (!isLoading) {
        // form에 id를 부여하고 해당 form을 직접 submit
        const form = e.currentTarget.form;
        if (form) {
          form.requestSubmit();
        }
      }
    }
  };

  return (
    <div className="flex flex-col items-center p-12 bg-gray-50 min-h-screen">
      <div className="w-full max-w-3xl p-8 space-y-6 bg-white rounded-xl shadow-md">
        <h1 className="text-2xl font-bold text-center text-gray-800">AI에게 질문하기</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="데이터베이스에 저장된 문서를 기반으로 질문을 입력하세요..."
            className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 text-black"
            rows={4}
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className="w-full px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400"
          >
            {isLoading ? "답변 생성 중..." : "질문하기"}
          </button>
        </form>

        {error && (
          <div className="mt-4 p-4 rounded-md text-sm bg-red-100 text-red-800">
            <p>{error}</p>
          </div>
        )}

        {/* 최종 답변을 표시 */}
        {answer && (
          <div className="mt-6">
            <h2 className="text-lg font-semibold text-gray-900">답변:</h2>
            <div className="mt-2 p-4 bg-gray-100 rounded-md">
              <p className="text-gray-800 whitespace-pre-wrap">{answer}</p>
            </div>
          </div>
        )}

        {/* 근거 문서를 목록으로 표시 */}
        {sources.length > 0 && (
          <div className="mt-6">
            <h3 className="text-md font-semibold text-gray-700">참고한 문서 조각:</h3>
            <ul className="mt-2 space-y-3">
              {sources.map((item) => (
                <li key={item.id} className="p-3 bg-gray-50 rounded-md border border-gray-200 text-sm">
                  <p className="text-gray-700 whitespace-pre-wrap truncate">"{item.content}"</p>
                  <div className="mt-2 text-xs text-gray-500">
                    <span>출처: {item.filename} (청크 #{item.chunk})</span>
                    <span className="ml-4 font-semibold">유사도: {item.similarity}</span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}