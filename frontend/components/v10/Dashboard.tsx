'use client'

import { TrendingUp, Users, Ticket, DollarSign, Activity, BarChart3 } from 'lucide-react'

export default function Dashboard() {
  const stats = [
    {
      title: '총 매출',
      value: '₩12,450,000',
      change: '+12.5%',
      trend: 'up',
      icon: DollarSign,
      color: 'text-green-600'
    },
    {
      title: '활성 멤버',
      value: '1,234',
      change: '+8.2%',
      trend: 'up',
      icon: Users,
      color: 'text-blue-600'
    },
    {
      title: '예매된 티켓',
      value: '5,678',
      change: '+15.3%',
      trend: 'up',
      icon: Ticket,
      color: 'text-purple-600'
    },
    {
      title: '활성 베팅',
      value: '89',
      change: '-2.1%',
      trend: 'down',
      icon: Activity,
      color: 'text-orange-600'
    }
  ]

  const recentActivities = [
    { type: 'ticket', message: '홍길동님이 맨체스터 유나이티드 vs 리버풀 경기 티켓을 예매했습니다', time: '5분 전' },
    { type: 'bet', message: '김철수님이 바르셀로나 승리 베팅을 완료했습니다', time: '12분 전' },
    { type: 'product', message: '프리미엄 시즌권이 새로 등록되었습니다', time: '1시간 전' },
    { type: 'member', message: '이영희님이 VIP 멤버로 승급했습니다', time: '2시간 전' },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-800">대시보드</h2>
        <p className="text-gray-600 mt-1">시스템 전체 현황을 한눈에 확인하세요</p>
      </div>

      {/* 통계 카드 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, index) => {
          const Icon = stat.icon
          return (
            <div key={index} className="bg-white p-6 rounded-lg shadow">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">{stat.title}</p>
                  <p className="text-2xl font-bold mt-2">{stat.value}</p>
                  <div className="flex items-center gap-1 mt-2">
                    {stat.trend === 'up' ? (
                      <TrendingUp className="w-4 h-4 text-green-600" />
                    ) : (
                      <TrendingUp className="w-4 h-4 text-red-600 rotate-180" />
                    )}
                    <span className={`text-sm font-medium ${
                      stat.trend === 'up' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {stat.change}
                    </span>
                    <span className="text-sm text-gray-500">전월 대비</span>
                  </div>
                </div>
                <div className={`p-3 rounded-full bg-gray-100 ${stat.color}`}>
                  <Icon className="w-6 h-6" />
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 최근 활동 */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">최근 활동</h3>
            <Activity className="w-5 h-5 text-gray-400" />
          </div>
          <div className="space-y-4">
            {recentActivities.map((activity, index) => (
              <div key={index} className="flex items-start gap-3 pb-4 border-b border-gray-200 last:border-0">
                <div className="w-2 h-2 rounded-full bg-blue-600 mt-2"></div>
                <div className="flex-1">
                  <p className="text-sm text-gray-800">{activity.message}</p>
                  <p className="text-xs text-gray-500 mt-1">{activity.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 빠른 액세스 */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">빠른 액세스</h3>
            <BarChart3 className="w-5 h-5 text-gray-400" />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-blue-50 rounded-lg cursor-pointer hover:bg-blue-100 transition-colors">
              <div className="text-sm font-medium text-blue-900">경기 추가</div>
              <div className="text-xs text-blue-700 mt-1">새로운 경기 등록</div>
            </div>
            <div className="p-4 bg-green-50 rounded-lg cursor-pointer hover:bg-green-100 transition-colors">
              <div className="text-sm font-medium text-green-900">베팅 분석</div>
              <div className="text-xs text-green-700 mt-1">승률 재계산</div>
            </div>
            <div className="p-4 bg-purple-50 rounded-lg cursor-pointer hover:bg-purple-100 transition-colors">
              <div className="text-sm font-medium text-purple-900">상품 관리</div>
              <div className="text-xs text-purple-700 mt-1">상품 추가/수정</div>
            </div>
            <div className="p-4 bg-orange-50 rounded-lg cursor-pointer hover:bg-orange-100 transition-colors">
              <div className="text-sm font-medium text-orange-900">멤버 관리</div>
              <div className="text-xs text-orange-700 mt-1">멤버 추가/수정</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

