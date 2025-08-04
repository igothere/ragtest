// src/app/page.tsx
import { redirect } from 'next/navigation';

export default function RootPage() {
  // 사용자가 루트 URL('/')에 접속하면 '/upload'로 즉시 리디렉션합니다.
  redirect('/chat');

  // redirect()는 예외를 발생시키므로 아래 코드는 실행되지 않습니다.
  return null;
}