'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Menu, X, ArrowLeft } from 'lucide-react'
import { cn } from '@/lib/utils'

/**
 * V1 Main 페이지 컴포넌트
 *
 * @remarks
 * - SHADCN_POLICY에 따라 하나의 컴포넌트로 모바일/데스크톱 모두 처리
 * - 모바일: 햄버거 메뉴 → Sheet 스타일 사이드 메뉴
 * - 데스크톱: 가로 네비게이션 메뉴
 * - Tailwind CSS의 responsive variant만 사용하여 반응형 처리
 * - 접근성(a11y)과 hydration 에러를 고려한 구현
 */
export default function V1Main() {
    const [isMenuOpen, setIsMenuOpen] = useState(false)

    const menuItems = [
        { href: '/v1/main', label: 'V1 Main', active: true },
        { href: '/v1/admin', label: 'V1 Admin' },
        { href: '/v10/main', label: 'V10 Main' },
        { href: '/v10/admin', label: 'V10 Admin' },
    ]

    return (
        <main className="min-h-screen bg-white text-black">
            {/* 헤더 - 모바일: 햄버거 메뉴, 데스크톱: 가로 네비게이션 */}
            <header className="sticky top-0 z-40 w-full border-b border-gray-200 bg-white">
                <div className="container mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex h-16 items-center justify-between">
                        {/* 왼쪽: 뒤로가기 + 제목 - 모바일: 작게, 데스크톱: 보통 크기 */}
                        <div className="flex items-center gap-3">
                            <Link
                                href="/"
                                className="p-2 text-black hover:bg-gray-100 rounded-lg transition-colors"
                                aria-label="홈으로 돌아가기"
                            >
                                <ArrowLeft className="w-5 h-5" />
                            </Link>
                            <h1 className="text-lg font-bold sm:text-xl lg:text-2xl">
                                V1 Main
                            </h1>
                        </div>

                        {/* 데스크톱 네비게이션 - lg 이상에서만 표시 */}
                        <nav className="hidden lg:flex lg:items-center lg:gap-6">
                            {menuItems.map((item) => (
                                <Link
                                    key={item.href}
                                    href={item.href}
                                    className={cn(
                                        'text-sm font-medium transition-colors px-3 py-2 rounded-lg',
                                        item.active
                                            ? 'text-black bg-gray-100'
                                            : 'text-gray-600 hover:text-black hover:bg-gray-50'
                                    )}
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
                                            className={cn(
                                                'block px-4 py-3 rounded-lg transition-colors font-medium',
                                                item.active
                                                    ? 'text-black bg-gray-100'
                                                    : 'text-gray-600 hover:text-black hover:bg-gray-50'
                                            )}
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

            {/* 메인 컨텐츠 - 반응형 레이아웃 */}
            <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 lg:py-16">
                <div className="max-w-4xl mx-auto">
                    {/* 제목 섹션 - 모바일: 작게, 데스크톱: 크게 */}
                    <div className="text-center mb-8 sm:mb-12 lg:mb-16">
                        <h2 className="text-3xl font-bold text-black sm:text-4xl lg:text-5xl mb-4">
                            V1 Main 페이지
                        </h2>
                        <p className="text-gray-600 text-sm sm:text-base lg:text-lg">
                            개인 프로젝트 메인 페이지입니다
                        </p>
                    </div>

                    {/* 카드 그리드 - 모바일: 1열, 태블릿: 2열, 데스크톱: 3열 */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
                        {[1, 2, 3, 4, 5, 6].map((item) => (
                            <div
                                key={item}
                                className="bg-gray-50 border border-gray-200 rounded-lg p-4 sm:p-6 hover:shadow-md transition-shadow"
                            >
                                <h3 className="text-lg font-semibold text-black mb-2">
                                    카드 {item}
                                </h3>
                                <p className="text-sm text-gray-600">
                                    카드 내용 설명입니다. 반응형 레이아웃으로 자동 조정됩니다.
                                </p>
                            </div>
                        ))}
                    </div>

                    {/* 추가 컨텐츠 섹션 - 모바일: 세로, 데스크톱: 가로 */}
                    <div className="mt-8 sm:mt-12 lg:mt-16 grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
                        <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
                            <h3 className="text-xl font-semibold text-black mb-4">
                                왼쪽 섹션
                            </h3>
                            <p className="text-gray-600 text-sm sm:text-base">
                                모바일에서는 세로로, 데스크톱에서는 가로로 배치됩니다.
                            </p>
                        </div>
                        <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
                            <h3 className="text-xl font-semibold text-black mb-4">
                                오른쪽 섹션
                            </h3>
                            <p className="text-gray-600 text-sm sm:text-base">
                                Tailwind CSS의 responsive variant로 자동 조정됩니다.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    )
}

