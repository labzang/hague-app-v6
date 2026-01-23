'use client'

import { useState } from 'react'
import {
  LayoutDashboard,
  Ticket,
  TrendingUp,
  Package,
  Users,
  Menu,
  X
} from 'lucide-react'

type TabType = 'dashboard' | 'matches' | 'betting' | 'products' | 'members'

interface AdminLayoutProps {
  children: React.ReactNode
  activeTab: TabType
  onTabChange: (tab: TabType) => void
}

export default function AdminLayout({ children, activeTab, onTabChange }: AdminLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const tabs = [
    { id: 'dashboard' as TabType, label: '대시보드', icon: LayoutDashboard },
    { id: 'matches' as TabType, label: '경기 표 예매', icon: Ticket },
    { id: 'betting' as TabType, label: '베팅 시스템', icon: TrendingUp },
    { id: 'products' as TabType, label: '상품 관리', icon: Package },
    { id: 'members' as TabType, label: '멤버 관리', icon: Users },
  ]

  return (
    <div className="flex h-screen bg-gray-50">
      {/* 사이드바 */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 bg-gray-900 text-white overflow-hidden`}>
        <div className="p-6 border-b border-gray-800">
          <h1 className="text-xl font-bold">어드민 대시보드</h1>
        </div>
        <nav className="mt-6">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`w-full flex items-center gap-3 px-6 py-3 text-left transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{tab.label}</span>
              </button>
            )
          })}
        </nav>
      </div>

      {/* 메인 컨텐츠 */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* 헤더 */}
        <header className="bg-white shadow-sm border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg hover:bg-gray-100"
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-600">관리자</span>
            </div>
          </div>
        </header>

        {/* 컨텐츠 영역 */}
        <main className="flex-1 overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  )
}

