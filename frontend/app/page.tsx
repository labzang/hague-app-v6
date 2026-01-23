'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Menu, X } from 'lucide-react'
import { cn } from '@/lib/utils'

/**
 * 메인 홈 페이지 컴포넌트
 *
 * @remarks
 * - SHADCN_POLICY에 따라 하나의 컴포넌트로 모바일/데스크톱 모두 처리
 * - 모바일: 햄버거 메뉴 → Sheet 스타일 사이드 메뉴
 * - 데스크톱: 가로 네비게이션 메뉴
 * - Tailwind CSS의 responsive variant만 사용하여 반응형 처리
 */
export default function Home() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  const menuItems = [
    { href: '/v1/main', label: 'V1 Main' },
    { href: '/v1/admin', label: 'V1 Admin' },
    { href: '/v10/main', label: 'V10 Main' },
    { href: '/v10/admin', label: 'V10 Admin' },
  ]

  return (
    <main className="min-h-screen bg-white text-black">
      {/* 헤더 - 모바일: 햄버거 메뉴, 데스크톱: 가로 네비게이션 */}
      <header className="sticky top-0 z-40 w-full border-b border-gray-200 bg-white">
        <div className="container mx-auto px-4">
          <div className="flex h-16 items-center justify-between">
            {/* 로고/제목 - 모바일: 작게, 데스크톱: 보통 크기 */}
            <h1 className="text-lg font-bold lg:text-xl">랩장 프로그램</h1>

            {/* 데스크톱 네비게이션 - lg 이상에서만 표시 */}
            <nav className="hidden lg:flex lg:items-center lg:gap-6">
              {menuItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="text-sm font-medium text-black transition-colors hover:text-gray-600"
                >
                  {item.label}
                </Link>
              ))}
            </nav>

            {/* 모바일 햄버거 메뉴 버튼 - lg 미만에서만 표시 */}
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="lg:hidden p-2 text-black hover:bg-gray-100 rounded-lg transition-colors"
              aria-label={isMenuOpen ? '메뉴 닫기' : '메뉴 열기'}
              aria-expanded={isMenuOpen}
            >
              {isMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>
      </header>

      {/* 모바일 사이드 메뉴 - Sheet 스타일 */}
      {isMenuOpen && (
        <>
          {/* 오버레이 - 모바일에서만 표시 */}
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
            onClick={() => setIsMenuOpen(false)}
            aria-hidden="true"
          />

          {/* 사이드 메뉴 패널 - 모바일: 왼쪽에서 슬라이드, 데스크톱: 숨김 */}
          <div
            className={cn(
              'fixed top-0 left-0 h-full w-64 bg-white shadow-lg z-50',
              'transform transition-transform duration-300 ease-in-out',
              'lg:hidden',
              isMenuOpen ? 'translate-x-0' : '-translate-x-full'
            )}
            role="dialog"
            aria-modal="true"
            aria-label="메뉴"
          >
            {/* 메뉴 헤더 */}
            <div className="flex h-16 items-center justify-between border-b border-gray-200 px-4">
              <h2 className="text-xl font-bold">메뉴</h2>
              <button
                onClick={() => setIsMenuOpen(false)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                aria-label="메뉴 닫기"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* 메뉴 목록 */}
            <nav className="p-4">
              <ul className="space-y-2">
                {menuItems.map((item) => (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      onClick={() => setIsMenuOpen(false)}
                      className="block px-4 py-3 text-black hover:bg-gray-100 rounded-lg transition-colors font-medium"
                    >
                      {item.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </nav>
          </div>
        </>
      )}

      {/* 메인 컨텐츠 - 가운데 정렬 */}
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4">
        <div className="text-center">
          {/* 모바일: 작은 크기, 데스크톱: 큰 크기 */}
          <h2 className="text-3xl font-bold text-black sm:text-4xl lg:text-5xl">
            랩장 프로그램
          </h2>
          <p className="mt-4 text-gray-600 text-sm sm:text-base lg:text-lg">
            축구 경기 표 예매, 베팅 시스템, 상품 및 멤버 관리
          </p>
        </div>
      </div>
    </main>
  )
}
