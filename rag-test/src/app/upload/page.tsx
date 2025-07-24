"use client";

import { useState, ChangeEvent, FormEvent } from "react";
import axios from "axios";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"initial" | "uploading" | "success" | "error">("initial");
  const [message, setMessage] = useState<string>("");

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
      setStatus("initial");
      setMessage("");
    }
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) {
      setMessage("파일을 선택해주세요.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setStatus("uploading");
    setMessage("파일을 업로드하고 처리 중입니다... 잠시만 기다려주세요.");

    try {
      // Flask API 서버 주소로 요청
      const response = await axios.post("http://localhost:5001/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        timeout: 300000, // 5분 타임아웃
      });
      
      setStatus("success");
      setMessage(`성공: ${response.data.message}`);
      console.log("Server response:", response.data);

    } catch (error: any) {
      setStatus("error");
      const errorMsg = error.response?.data?.error || "알 수 없는 오류가 발생했습니다.";
      const errorDetails = error.response?.data?.details || "";
      setMessage(`실패: ${errorMsg}\n${errorDetails}`);
      console.error("Upload error:", error);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-gray-50">
      <div className="w-full max-w-lg p-8 space-y-6 bg-white rounded-xl shadow-md">
        <h1 className="text-2xl font-bold text-center text-gray-800">문서 업로드 및 벡터화</h1>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="file-upload" className="block text-sm font-medium text-gray-700">
              PDF, TXT, MD 파일 선택
            </label>
            <input
              id="file-upload"
              type="file"
              onChange={handleFileChange}
              accept=".pdf,.txt,.md"
              className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>
          <button
            type="submit"
            disabled={!file || status === "uploading"}
            className="w-full px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400"
          >
            {status === "uploading" ? "처리 중..." : "업로드 및 처리 시작"}
          </button>
        </form>
        {message && (
          <div className={`mt-4 p-4 rounded-md text-sm ${
              status === 'success' ? 'bg-green-100 text-green-800' : ''
            } ${
              status === 'error' ? 'bg-red-100 text-red-800' : ''
            } ${
              status === 'uploading' ? 'bg-yellow-100 text-yellow-800' : ''
            }`}
          >
            <pre className="whitespace-pre-wrap">{message}</pre>
          </div>
        )}
      </div>
    </main>
  );
}