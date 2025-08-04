// src/components/Navigation.tsx
"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navigation() {
  const pathname = usePathname();

  const navItems = [
    { href: '/upload', label: '문서 업로드' },
    { href: '/chat', label: 'AI에게 질문하기' },
  ];

  return (
    <nav className="bg-white shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-center h-16">
          <div className="flex items-center">
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-4">
                {navItems.map((item) => (
                  <Link
                    key={item.label}
                    href={item.href}
                    className={`${
                      pathname === item.href
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-600 hover:bg-gray-200 hover:text-black'
                    } px-3 py-2 rounded-md text-sm font-medium`}
                  >
                    {item.label}
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}