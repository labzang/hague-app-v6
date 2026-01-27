'use client'

import { useState } from 'react'
import { LayoutDashboard, Ticket, TrendingUp, Package, Users, ArrowLeft, Menu, X, Upload } from 'lucide-react'
import Link from 'next/link'
import { cn } from '@/lib/utils'

type TabType = 'dashboard' | 'matches' | 'betting' | 'products' | 'members'

/**
 * V10 Admin 페이지 컴포넌트 (어드민 대시보드)
 *
 * @remarks
 * - SHADCN_POLICY에 따라 하나의 컴포넌트로 모바일/데스크톱 모두 처리
 * - 모바일: 햄버거 메뉴 → Sheet 스타일 사이드 메뉴
 * - 데스크톱: 고정 사이드바 + 메인 컨텐츠
 * - Tailwind CSS의 responsive variant만 사용하여 반응형 처리
 * - 접근성(a11y)과 hydration 에러를 고려한 구현
 */
export default function V10Admin() {
    const [activeTab, setActiveTab] = useState<TabType>('dashboard')
    const [sidebarOpen, setSidebarOpen] = useState(false)

    const tabs = [
        { id: 'dashboard' as TabType, label: '대시보드', icon: LayoutDashboard },
        { id: 'matches' as TabType, label: '경기 예매', icon: Ticket },
        { id: 'betting' as TabType, label: '베팅 시스템', icon: TrendingUp },
        { id: 'products' as TabType, label: '상품 관리', icon: Package },
        { id: 'members' as TabType, label: '멤버 관리', icon: Users },
    ]

    const renderContent = () => {
        switch (activeTab) {
            case 'dashboard':
                return <DashboardView />
            case 'matches':
                return <MatchesView />
            case 'betting':
                return <BettingView />
            case 'products':
                return <ProductsView />
            case 'members':
                return <MembersView />
            default:
                return <DashboardView />
        }
    }

    return (
        <div className="min-h-screen bg-white text-black flex">
            {/* 사이드바 - 데스크톱: 항상 보임, 모바일: 토글 가능 */}
            <aside
                className={cn(
                    'bg-gray-900 text-white transition-all duration-300 overflow-hidden',
                    'fixed lg:static h-screen z-50 lg:z-auto',
                    // 모바일: 토글 가능, 데스크톱: 항상 표시
                    sidebarOpen ? 'w-64' : 'w-0 lg:w-64'
                )}
            >
                <div className="p-6 border-b border-gray-800">
                    <div className="flex items-center justify-between">
                        <h1 className="text-xl font-bold">어드민 대시보드</h1>
                        {/* 모바일에서만 닫기 버튼 표시 */}
                        <button
                            onClick={() => setSidebarOpen(false)}
                            className="lg:hidden p-2 hover:bg-gray-800 rounded-lg transition-colors"
                            aria-label="사이드바 닫기"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>
                </div>
                <nav className="mt-6 flex flex-col h-[calc(100vh-120px)]">
                    <div className="flex-1">
                        {tabs.map((tab) => {
                            const Icon = tab.icon
                            return (
                                <button
                                    key={tab.id}
                                    onClick={() => {
                                        setActiveTab(tab.id)
                                        // 모바일에서 탭 선택 시 사이드바 자동 닫기
                                        setSidebarOpen(false)
                                    }}
                                    className={cn(
                                        'w-full flex items-center gap-3 px-6 py-3 text-left transition-colors',
                                        activeTab === tab.id
                                            ? 'bg-blue-600 text-white'
                                            : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                                    )}
                                >
                                    <Icon className="w-5 h-5" />
                                    <span>{tab.label}</span>
                                </button>
                            )
                        })}
                    </div>
                    {/* 파일 업로드 메뉴 - 맨 아래 고정 */}
                    <div className="border-t border-gray-800 pt-2">
                        <Link
                            href="/v10/admin/upload/player"
                            onClick={() => setSidebarOpen(false)}
                            className={cn(
                                'w-full flex items-center gap-3 px-6 py-3 text-left transition-colors',
                                'text-gray-300 hover:bg-gray-800 hover:text-white'
                            )}
                        >
                            <Upload className="w-5 h-5" />
                            <span>파일 업로드</span>
                        </Link>
                    </div>
                </nav>
            </aside>

            {/* 메인 컨텐츠 영역 */}
            <div className="flex-1 flex flex-col min-w-0">
                {/* 헤더 - 모바일: 햄버거 메뉴, 데스크톱: 제목만 */}
                <header className="sticky top-0 z-30 bg-white border-b border-gray-200 px-4 sm:px-6 lg:px-8 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            {/* 모바일 햄버거 메뉴 버튼 - lg 미만에서만 표시 */}
                            <button
                                onClick={() => setSidebarOpen(!sidebarOpen)}
                                className="lg:hidden p-2 hover:bg-gray-100 rounded-lg transition-colors"
                                aria-label={sidebarOpen ? '메뉴 닫기' : '메뉴 열기'}
                                aria-expanded={sidebarOpen}
                            >
                                <Menu className="w-6 h-6" />
                            </button>
                            <Link
                                href="/"
                                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                                aria-label="홈으로 돌아가기"
                            >
                                <ArrowLeft className="w-5 h-5" />
                            </Link>
                            {/* 모바일: 작게, 데스크톱: 보통 크기 */}
                            <h1 className="text-lg font-bold sm:text-xl lg:text-2xl">
                                어드민 관리
                            </h1>
                        </div>
                        <Link
                            href="http://localhost:3000"
                            className="text-sm text-gray-600 hidden sm:block hover:text-gray-900 transition-colors"
                        >
                            로그아웃
                        </Link>
                    </div>
                </header>

                {/* 컨텐츠 - 반응형 패딩 */}
                <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8">
                    {renderContent()}
                </main>
            </div>

            {/* 모바일 사이드바 오버레이 - lg 미만에서만 표시 */}
            {sidebarOpen && (
                <div
                    className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-40"
                    onClick={() => setSidebarOpen(false)}
                    aria-hidden="true"
                />
            )}
        </div>
    )
}

/**
 * 대시보드 뷰 컴포넌트
 * 모바일: 1-2열, 데스크톱: 4열 그리드
 */
function DashboardView() {
    return (
        <div className="space-y-6">
            <div>
                {/* 모바일: 작게, 데스크톱: 크게 */}
                <h2 className="text-2xl font-bold mb-2 sm:text-3xl lg:text-4xl">대시보드</h2>
                <p className="text-gray-600 text-sm sm:text-base">
                    시스템 전체 현황을 한눈에 확인하세요
                </p>
            </div>

            {/* 통계 카드 - 모바일: 1열, 태블릿: 2열, 데스크톱: 4열 */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
                <div className="bg-gray-50 p-4 sm:p-6 rounded-lg border border-gray-200">
                    <div className="text-sm text-gray-600 mb-2">총 매출</div>
                    <div className="text-2xl font-bold mb-2 sm:text-3xl">₩12,450,000</div>
                    <div className="text-sm text-green-600">+12.5% 전월 대비</div>
                </div>
                <div className="bg-gray-50 p-4 sm:p-6 rounded-lg border border-gray-200">
                    <div className="text-sm text-gray-600 mb-2">활성 멤버</div>
                    <div className="text-2xl font-bold mb-2 sm:text-3xl">1,234</div>
                    <div className="text-sm text-green-600">+8.2% 전월 대비</div>
                </div>
                <div className="bg-gray-50 p-4 sm:p-6 rounded-lg border border-gray-200">
                    <div className="text-sm text-gray-600 mb-2">예매 티켓</div>
                    <div className="text-2xl font-bold mb-2 sm:text-3xl">5,678</div>
                    <div className="text-sm text-green-600">+15.3% 전월 대비</div>
                </div>
                <div className="bg-gray-50 p-4 sm:p-6 rounded-lg border border-gray-200">
                    <div className="text-sm text-gray-600 mb-2">활성 베팅</div>
                    <div className="text-2xl font-bold mb-2 sm:text-3xl">89</div>
                    <div className="text-sm text-red-600">-2.1% 전월 대비</div>
                </div>
            </div>

            {/* 최근 활동과 빠른 액세스 - 모바일: 1열, 데스크톱: 2열 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-gray-50 p-4 sm:p-6 rounded-lg border border-gray-200">
                    <h3 className="text-lg font-semibold mb-4">최근 활동</h3>
                    <div className="space-y-4">
                        <div className="pb-4 border-b border-gray-200 last:border-0">
                            <div className="font-medium text-sm">홍길동님이 맨체스터 유나이티드 vs 리버풀 경기 티켓을 예매했습니다</div>
                            <div className="text-gray-500 text-xs mt-1">5분 전</div>
                        </div>
                        <div className="pb-4 border-b border-gray-200 last:border-0">
                            <div className="font-medium text-sm">김철수님이 바르셀로나 승리 베팅을 완료했습니다</div>
                            <div className="text-gray-500 text-xs mt-1">12분 전</div>
                        </div>
                        <div className="pb-4 border-b border-gray-200 last:border-0">
                            <div className="font-medium text-sm">프리미엄 시즌권이 새로 등록되었습니다</div>
                            <div className="text-gray-500 text-xs mt-1">1시간 전</div>
                        </div>
                        <div>
                            <div className="font-medium text-sm">이영희님이 VIP 멤버로 승급했습니다</div>
                            <div className="text-gray-500 text-xs mt-1">2시간 전</div>
                        </div>
                    </div>
                </div>

                <div className="bg-gray-50 p-4 sm:p-6 rounded-lg border border-gray-200">
                    <h3 className="text-lg font-semibold mb-4">빠른 액세스</h3>
                    {/* 모바일: 1열, 데스크톱: 2열 */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <button className="p-4 bg-blue-50 hover:bg-blue-100 rounded-lg text-left transition-colors">
                            <div className="font-medium text-blue-900 text-sm">경기 추가</div>
                            <div className="text-xs text-blue-700 mt-1">새로운 경기 등록</div>
                        </button>
                        <button className="p-4 bg-green-50 hover:bg-green-100 rounded-lg text-left transition-colors">
                            <div className="font-medium text-green-900 text-sm">베팅 분석</div>
                            <div className="text-xs text-green-700 mt-1">승률 재계산</div>
                        </button>
                        <button className="p-4 bg-purple-50 hover:bg-purple-100 rounded-lg text-left transition-colors">
                            <div className="font-medium text-purple-900 text-sm">상품 관리</div>
                            <div className="text-xs text-purple-700 mt-1">상품 추가/수정</div>
                        </button>
                        <button className="p-4 bg-orange-50 hover:bg-orange-100 rounded-lg text-left transition-colors">
                            <div className="font-medium text-orange-900 text-sm">멤버 관리</div>
                            <div className="text-xs text-orange-700 mt-1">멤버 추가/수정</div>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}

/**
 * 경기 예매 관리 뷰
 * 모바일: 카드 형식, 데스크톱: 테이블 형식
 */
function MatchesView() {
    return (
        <div className="space-y-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h2 className="text-2xl font-bold mb-2 sm:text-3xl lg:text-4xl">경기 예매 관리</h2>
                    <p className="text-gray-600 text-sm sm:text-base">축구 경기 표 예매를 관리합니다</p>
                </div>
                <button className="w-full sm:w-auto px-6 py-3 bg-black text-white rounded-lg hover:bg-gray-800 transition-colors font-medium text-sm sm:text-base">
                    경기 추가
                </button>
            </div>

            {/* 모바일: 카드 형식, 데스크톱: 테이블 형식 */}
            <div className="lg:hidden space-y-4">
                {/* 모바일 카드 뷰 */}
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="font-medium text-lg mb-2">맨체스터 유나이티드 vs 리버풀</div>
                    <div className="text-sm text-gray-600 space-y-1">
                        <div>2024-03-15 20:00</div>
                        <div>올드 트래퍼드</div>
                        <div>예매 가능: 5,000매</div>
                        <div className="font-medium">₩50,000</div>
                    </div>
                    <div className="mt-3 flex gap-2">
                        <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">예정</span>
                        <button className="text-blue-600 hover:text-blue-800 text-sm">수정</button>
                    </div>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="font-medium text-lg mb-2">바르셀로나 vs 레알 마드리드</div>
                    <div className="text-sm text-gray-600 space-y-1">
                        <div>2024-03-20 22:00</div>
                        <div>캄프 누</div>
                        <div>예매 가능: 3,000매</div>
                        <div className="font-medium">₩80,000</div>
                    </div>
                    <div className="mt-3 flex gap-2">
                        <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">예정</span>
                        <button className="text-blue-600 hover:text-blue-800 text-sm">수정</button>
                    </div>
                </div>
            </div>

            {/* 데스크톱 테이블 뷰 */}
            <div className="hidden lg:block bg-white border border-gray-200 rounded-lg overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">경기</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">일시</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">장소</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">예매 가능</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">가격</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">상태</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">작업</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            <tr className="hover:bg-gray-50">
                                <td className="px-6 py-4">
                                    <div className="font-medium">맨체스터 유나이티드</div>
                                    <div className="text-sm text-gray-500">vs 리버풀</div>
                                </td>
                                <td className="px-6 py-4 text-sm">
                                    <div>2024-03-15</div>
                                    <div className="text-gray-500">20:00</div>
                                </td>
                                <td className="px-6 py-4 text-sm">올드 트래퍼드</td>
                                <td className="px-6 py-4 text-sm">5,000매</td>
                                <td className="px-6 py-4 text-sm font-medium">₩50,000</td>
                                <td className="px-6 py-4">
                                    <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">예정</span>
                                </td>
                                <td className="px-6 py-4">
                                    <button className="text-blue-600 hover:text-blue-800 text-sm">수정</button>
                                </td>
                            </tr>
                            <tr className="hover:bg-gray-50">
                                <td className="px-6 py-4">
                                    <div className="font-medium">바르셀로나</div>
                                    <div className="text-sm text-gray-500">vs 레알 마드리드</div>
                                </td>
                                <td className="px-6 py-4 text-sm">
                                    <div>2024-03-20</div>
                                    <div className="text-gray-500">22:00</div>
                                </td>
                                <td className="px-6 py-4 text-sm">캄프 누</td>
                                <td className="px-6 py-4 text-sm">3,000매</td>
                                <td className="px-6 py-4 text-sm font-medium">₩80,000</td>
                                <td className="px-6 py-4">
                                    <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">예정</span>
                                </td>
                                <td className="px-6 py-4">
                                    <button className="text-blue-600 hover:text-blue-800 text-sm">수정</button>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}

/**
 * 베팅 시스템 뷰
 * 모바일: 1열, 데스크톱: 2열 그리드
 */
function BettingView() {
    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-2xl font-bold mb-2 sm:text-3xl lg:text-4xl">베팅 시스템</h2>
                <p className="text-gray-600 text-sm sm:text-base">승률 추론 기반 베팅 옵션을 관리합니다</p>
            </div>

            {/* 모바일: 1열, 데스크톱: 2열 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="border border-gray-200 rounded-lg p-4 sm:p-6">
                    <div className="font-semibold text-lg mb-4">맨체스터 유나이티드 vs 리버풀</div>
                    <div className="grid grid-cols-3 gap-2 sm:gap-4 mb-4">
                        <div className="text-center p-3 sm:p-4 bg-gray-50 rounded-lg">
                            <div className="text-xs text-gray-600 mb-2">홈 승</div>
                            <div className="text-xl sm:text-2xl font-bold mb-1">35%</div>
                            <div className="text-sm text-blue-600 font-medium">2.86x</div>
                        </div>
                        <div className="text-center p-3 sm:p-4 bg-gray-50 rounded-lg">
                            <div className="text-xs text-gray-600 mb-2">무승부</div>
                            <div className="text-xl sm:text-2xl font-bold mb-1">25%</div>
                            <div className="text-sm text-blue-600 font-medium">4.0x</div>
                        </div>
                        <div className="text-center p-3 sm:p-4 bg-gray-50 rounded-lg">
                            <div className="text-xs text-gray-600 mb-2">원정 승</div>
                            <div className="text-xl sm:text-2xl font-bold mb-1">40%</div>
                            <div className="text-sm text-blue-600 font-medium">2.5x</div>
                        </div>
                    </div>
                    <div className="flex justify-between items-center text-sm">
                        <span className="text-gray-500">신뢰도: 85%</span>
                        <button className="text-blue-600 hover:text-blue-800">재계산</button>
                    </div>
                </div>

                <div className="border border-gray-200 rounded-lg p-4 sm:p-6">
                    <div className="font-semibold text-lg mb-4">바르셀로나 vs 레알 마드리드</div>
                    <div className="grid grid-cols-3 gap-2 sm:gap-4 mb-4">
                        <div className="text-center p-3 sm:p-4 bg-gray-50 rounded-lg">
                            <div className="text-xs text-gray-600 mb-2">홈 승</div>
                            <div className="text-xl sm:text-2xl font-bold mb-1">45%</div>
                            <div className="text-sm text-blue-600 font-medium">2.22x</div>
                        </div>
                        <div className="text-center p-3 sm:p-4 bg-gray-50 rounded-lg">
                            <div className="text-xs text-gray-600 mb-2">무승부</div>
                            <div className="text-xl sm:text-2xl font-bold mb-1">30%</div>
                            <div className="text-sm text-blue-600 font-medium">3.33x</div>
                        </div>
                        <div className="text-center p-3 sm:p-4 bg-gray-50 rounded-lg">
                            <div className="text-xs text-gray-600 mb-2">원정 승</div>
                            <div className="text-xl sm:text-2xl font-bold mb-1">25%</div>
                            <div className="text-sm text-blue-600 font-medium">4.0x</div>
                        </div>
                    </div>
                    <div className="flex justify-between items-center text-sm">
                        <span className="text-gray-500">신뢰도: 92%</span>
                        <button className="text-blue-600 hover:text-blue-800">재계산</button>
                    </div>
                </div>
            </div>
        </div>
    )
}

/**
 * 상품 관리 뷰
 * 모바일: 카드 형식, 데스크톱: 테이블 형식
 */
function ProductsView() {
    return (
        <div className="space-y-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h2 className="text-2xl font-bold mb-2 sm:text-3xl lg:text-4xl">상품 관리</h2>
                    <p className="text-gray-600 text-sm sm:text-base">파생 상품을 관리합니다</p>
                </div>
                <button className="w-full sm:w-auto px-6 py-3 bg-black text-white rounded-lg hover:bg-gray-800 transition-colors font-medium text-sm sm:text-base">
                    상품 추가
                </button>
            </div>

            {/* 모바일: 카드 형식 */}
            <div className="lg:hidden space-y-4">
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="font-medium text-lg mb-2">프리미엄 시즌권</div>
                    <div className="text-sm text-gray-600 mb-2">전 시즌 경기 무제한 관람</div>
                    <div className="flex items-center justify-between">
                        <div>
                            <div className="text-sm font-medium">₩500,000</div>
                            <div className="text-sm text-gray-600">재고: 50개</div>
                        </div>
                        <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">판매중</span>
                    </div>
                    <button className="mt-3 text-blue-600 hover:text-blue-800 text-sm">수정</button>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="font-medium text-lg mb-2">VIP 박스석 패키지</div>
                    <div className="text-sm text-gray-600 mb-2">VIP 박스석 + 식사 + 주차</div>
                    <div className="flex items-center justify-between">
                        <div>
                            <div className="text-sm font-medium">₩2,000,000</div>
                            <div className="text-sm text-gray-600">재고: 10개</div>
                        </div>
                        <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">판매중</span>
                    </div>
                    <button className="mt-3 text-blue-600 hover:text-blue-800 text-sm">수정</button>
                </div>
            </div>

            {/* 데스크톱 테이블 형식 */}
            <div className="hidden lg:block bg-white border border-gray-200 rounded-lg overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">상품명</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">카테고리</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">가격</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">재고</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">상태</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">작업</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            <tr className="hover:bg-gray-50">
                                <td className="px-6 py-4">
                                    <div className="font-medium">프리미엄 시즌권</div>
                                    <div className="text-sm text-gray-500">전 시즌 경기 무제한 관람</div>
                                </td>
                                <td className="px-6 py-4">
                                    <span className="px-2 py-1 text-xs font-medium bg-gray-100 text-gray-800 rounded">시즌권</span>
                                </td>
                                <td className="px-6 py-4 text-sm font-medium">₩500,000</td>
                                <td className="px-6 py-4 text-sm">50개</td>
                                <td className="px-6 py-4">
                                    <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">판매중</span>
                                </td>
                                <td className="px-6 py-4">
                                    <button className="text-blue-600 hover:text-blue-800 text-sm">수정</button>
                                </td>
                            </tr>
                            <tr className="hover:bg-gray-50">
                                <td className="px-6 py-4">
                                    <div className="font-medium">VIP 박스석 패키지</div>
                                    <div className="text-sm text-gray-500">VIP 박스석 + 식사 + 주차</div>
                                </td>
                                <td className="px-6 py-4">
                                    <span className="px-2 py-1 text-xs font-medium bg-gray-100 text-gray-800 rounded">패키지</span>
                                </td>
                                <td className="px-6 py-4 text-sm font-medium">₩2,000,000</td>
                                <td className="px-6 py-4 text-sm">10개</td>
                                <td className="px-6 py-4">
                                    <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">판매중</span>
                                </td>
                                <td className="px-6 py-4">
                                    <button className="text-blue-600 hover:text-blue-800 text-sm">수정</button>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}

/**
 * 멤버 관리 뷰
 * 모바일: 카드 형식, 데스크톱: 테이블 형식
 */
function MembersView() {
    return (
        <div className="space-y-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h2 className="text-2xl font-bold mb-2 sm:text-3xl lg:text-4xl">멤버 관리</h2>
                    <p className="text-gray-600 text-sm sm:text-base">멤버 정보를 관리합니다</p>
                </div>
                <button className="w-full sm:w-auto px-6 py-3 bg-black text-white rounded-lg hover:bg-gray-800 transition-colors font-medium text-sm sm:text-base">
                    멤버 추가
                </button>
            </div>

            {/* 모바일: 카드 형식 */}
            <div className="lg:hidden space-y-4">
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="font-medium text-lg mb-2">홍길동</div>
                    <div className="text-sm text-gray-600 mb-2">hong@example.com</div>
                    <div className="text-sm text-gray-600 mb-3">010-1234-5678</div>
                    <div className="flex items-center gap-2 mb-3">
                        <span className="px-2 py-1 text-xs font-medium bg-yellow-100 text-yellow-800 rounded-full">VIP</span>
                        <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">활성</span>
                    </div>
                    <div className="text-sm text-gray-600 mb-2">가입일: 2023-01-15</div>
                    <div className="text-sm font-medium mb-3">총 구매액: ₩2,500,000</div>
                    <button className="text-blue-600 hover:text-blue-800 text-sm">수정</button>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="font-medium text-lg mb-2">김철수</div>
                    <div className="text-sm text-gray-600 mb-2">kim@example.com</div>
                    <div className="text-sm text-gray-600 mb-3">010-2345-6789</div>
                    <div className="flex items-center gap-2 mb-3">
                        <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">일반</span>
                        <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">활성</span>
                    </div>
                    <div className="text-sm text-gray-600 mb-2">가입일: 2023-06-20</div>
                    <div className="text-sm font-medium mb-3">총 구매액: ₩800,000</div>
                    <button className="text-blue-600 hover:text-blue-800 text-sm">수정</button>
                </div>
            </div>

            {/* 데스크톱 테이블 형식 */}
            <div className="hidden lg:block bg-white border border-gray-200 rounded-lg overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">멤버</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">연락처</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">역할</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">상태</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">가입일</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">총 구매액</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">작업</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            <tr className="hover:bg-gray-50">
                                <td className="px-6 py-4">
                                    <div className="font-medium">홍길동</div>
                                    <div className="text-sm text-gray-500">hong@example.com</div>
                                </td>
                                <td className="px-6 py-4 text-sm">010-1234-5678</td>
                                <td className="px-6 py-4">
                                    <span className="px-2 py-1 text-xs font-medium bg-yellow-100 text-yellow-800 rounded-full">VIP</span>
                                </td>
                                <td className="px-6 py-4">
                                    <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">활성</span>
                                </td>
                                <td className="px-6 py-4 text-sm">2023-01-15</td>
                                <td className="px-6 py-4 text-sm font-medium">₩2,500,000</td>
                                <td className="px-6 py-4">
                                    <button className="text-blue-600 hover:text-blue-800 text-sm">수정</button>
                                </td>
                            </tr>
                            <tr className="hover:bg-gray-50">
                                <td className="px-6 py-4">
                                    <div className="font-medium">김철수</div>
                                    <div className="text-sm text-gray-500">kim@example.com</div>
                                </td>
                                <td className="px-6 py-4 text-sm">010-2345-6789</td>
                                <td className="px-6 py-4">
                                    <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">일반</span>
                                </td>
                                <td className="px-6 py-4">
                                    <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">활성</span>
                                </td>
                                <td className="px-6 py-4 text-sm">2023-06-20</td>
                                <td className="px-6 py-4 text-sm font-medium">₩800,000</td>
                                <td className="px-6 py-4">
                                    <button className="text-blue-600 hover:text-blue-800 text-sm">수정</button>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}
