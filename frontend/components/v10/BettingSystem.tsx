'use client'

import { useState } from 'react'
import { TrendingUp, TrendingDown, BarChart3, Calculator, RefreshCw } from 'lucide-react'

interface BettingOdds {
  id: string
  matchId: string
  homeTeam: string
  awayTeam: string
  homeWinProb: number
  drawProb: number
  awayWinProb: number
  homeOdds: number
  drawOdds: number
  awayOdds: number
  confidence: number
  lastUpdated: string
}

export default function BettingSystem() {
  const [odds, setOdds] = useState<BettingOdds[]>([
    {
      id: '1',
      matchId: '1',
      homeTeam: '맨체스터 유나이티드',
      awayTeam: '리버풀',
      homeWinProb: 35,
      drawProb: 25,
      awayWinProb: 40,
      homeOdds: 2.86,
      drawOdds: 4.0,
      awayOdds: 2.5,
      confidence: 85,
      lastUpdated: '2024-03-10 14:30'
    },
    {
      id: '2',
      matchId: '2',
      homeTeam: '바르셀로나',
      awayTeam: '레알 마드리드',
      homeWinProb: 45,
      drawProb: 30,
      awayWinProb: 25,
      homeOdds: 2.22,
      drawOdds: 3.33,
      awayOdds: 4.0,
      confidence: 92,
      lastUpdated: '2024-03-10 14:30'
    }
  ])

  const [selectedMatch, setSelectedMatch] = useState<BettingOdds | null>(null)
  const [betAmount, setBetAmount] = useState(10000)
  const [selectedOutcome, setSelectedOutcome] = useState<'home' | 'draw' | 'away' | null>(null)

  const calculatePayout = (odds: number, amount: number) => {
    return Math.round(odds * amount)
  }

  const calculateExpectedValue = (prob: number, odds: number, amount: number) => {
    return Math.round((prob / 100) * (odds * amount) - amount)
  }

  const recalculateOdds = (matchId: string) => {
    setOdds(odds.map(odd => {
      if (odd.id === matchId) {
        // 간단한 재계산 로직 (실제로는 ML 모델 사용)
        const newHomeProb = Math.min(100, Math.max(10, odd.homeWinProb + (Math.random() - 0.5) * 10))
        const newDrawProb = Math.min(100, Math.max(10, odd.drawProb + (Math.random() - 0.5) * 10))
        const newAwayProb = 100 - newHomeProb - newDrawProb

        return {
          ...odd,
          homeWinProb: Math.round(newHomeProb),
          drawProb: Math.round(newDrawProb),
          awayWinProb: Math.round(newAwayProb),
          homeOdds: Math.round((100 / newHomeProb) * 10) / 10,
          drawOdds: Math.round((100 / newDrawProb) * 10) / 10,
          awayOdds: Math.round((100 / newAwayProb) * 10) / 10,
          confidence: Math.min(100, odd.confidence + Math.round((Math.random() - 0.5) * 5)),
          lastUpdated: new Date().toLocaleString('ko-KR')
        }
      }
      return odd
    }))
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">승률 추론 기반 베팅 시스템</h2>
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <BarChart3 className="w-5 h-5" />
          <span>AI 기반 승률 분석</span>
        </div>
      </div>

      {/* 통계 카드 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-sm text-gray-600">활성 베팅</div>
          <div className="text-2xl font-bold mt-2">{odds.length}</div>
        </div>
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-sm text-gray-600">평균 신뢰도</div>
          <div className="text-2xl font-bold mt-2">
            {Math.round(odds.reduce((sum, o) => sum + o.confidence, 0) / odds.length || 0)}%
          </div>
        </div>
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-sm text-gray-600">최고 승률</div>
          <div className="text-2xl font-bold mt-2 text-green-600">
            {Math.max(...odds.map(o => Math.max(o.homeWinProb, o.drawProb, o.awayWinProb)))}%
          </div>
        </div>
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-sm text-gray-600">최저 배당</div>
          <div className="text-2xl font-bold mt-2 text-blue-600">
            {Math.min(...odds.map(o => Math.min(o.homeOdds, o.drawOdds, o.awayOdds))).toFixed(1)}x
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 베팅 옵션 목록 */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold">베팅 옵션</h3>
          </div>
          <div className="divide-y divide-gray-200">
            {odds.map((odd) => (
              <div
                key={odd.id}
                className={`p-4 cursor-pointer transition-colors ${
                  selectedMatch?.id === odd.id ? 'bg-blue-50' : 'hover:bg-gray-50'
                }`}
                onClick={() => setSelectedMatch(odd)}
              >
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <div className="font-medium text-gray-900">{odd.homeTeam}</div>
                    <div className="text-sm text-gray-500">vs {odd.awayTeam}</div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      recalculateOdds(odd.id)
                    }}
                    className="p-1 hover:bg-gray-200 rounded"
                    title="승률 재계산"
                  >
                    <RefreshCw className="w-4 h-4 text-gray-600" />
                  </button>
                </div>

                <div className="grid grid-cols-3 gap-2 mb-2">
                  <div className="text-center p-2 bg-gray-50 rounded">
                    <div className="text-xs text-gray-600">홈 승</div>
                    <div className="font-bold text-sm">{odd.homeWinProb}%</div>
                    <div className="text-xs text-blue-600">{odd.homeOdds}x</div>
                  </div>
                  <div className="text-center p-2 bg-gray-50 rounded">
                    <div className="text-xs text-gray-600">무승부</div>
                    <div className="font-bold text-sm">{odd.drawProb}%</div>
                    <div className="text-xs text-blue-600">{odd.drawOdds}x</div>
                  </div>
                  <div className="text-center p-2 bg-gray-50 rounded">
                    <div className="text-xs text-gray-600">원정 승</div>
                    <div className="font-bold text-sm">{odd.awayWinProb}%</div>
                    <div className="text-xs text-blue-600">{odd.awayOdds}x</div>
                  </div>
                </div>

                <div className="flex justify-between items-center text-xs">
                  <span className="text-gray-500">신뢰도: {odd.confidence}%</span>
                  <span className="text-gray-400">{odd.lastUpdated}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 베팅 계산기 */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">베팅 계산기</h3>

          {selectedMatch ? (
            <div className="space-y-4">
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="font-medium mb-1">{selectedMatch.homeTeam}</div>
                <div className="text-sm text-gray-600">vs {selectedMatch.awayTeam}</div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">베팅 금액 (원)</label>
                <input
                  type="number"
                  value={betAmount}
                  onChange={(e) => setBetAmount(parseInt(e.target.value) || 0)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  min="1000"
                  step="1000"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">결과 선택</label>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => setSelectedOutcome('home')}
                    className={`p-3 rounded-lg border-2 transition-colors ${
                      selectedOutcome === 'home'
                        ? 'border-blue-600 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-xs text-gray-600 mb-1">홈 승</div>
                    <div className="font-bold">{selectedMatch.homeOdds}x</div>
                    <div className="text-xs text-gray-500">{selectedMatch.homeWinProb}%</div>
                  </button>
                  <button
                    onClick={() => setSelectedOutcome('draw')}
                    className={`p-3 rounded-lg border-2 transition-colors ${
                      selectedOutcome === 'draw'
                        ? 'border-blue-600 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-xs text-gray-600 mb-1">무승부</div>
                    <div className="font-bold">{selectedMatch.drawOdds}x</div>
                    <div className="text-xs text-gray-500">{selectedMatch.drawProb}%</div>
                  </button>
                  <button
                    onClick={() => setSelectedOutcome('away')}
                    className={`p-3 rounded-lg border-2 transition-colors ${
                      selectedOutcome === 'away'
                        ? 'border-blue-600 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-xs text-gray-600 mb-1">원정 승</div>
                    <div className="font-bold">{selectedMatch.awayOdds}x</div>
                    <div className="text-xs text-gray-500">{selectedMatch.awayWinProb}%</div>
                  </button>
                </div>
              </div>

              {selectedOutcome && (
                <div className="p-4 bg-blue-50 rounded-lg space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-600">예상 배당금:</span>
                    <span className="font-bold text-lg">
                      ₩{calculatePayout(
                        selectedOutcome === 'home' ? selectedMatch.homeOdds :
                        selectedOutcome === 'draw' ? selectedMatch.drawOdds :
                        selectedMatch.awayOdds,
                        betAmount
                      ).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">예상 수익:</span>
                    <span className="font-bold text-green-600">
                      ₩{(calculatePayout(
                        selectedOutcome === 'home' ? selectedMatch.homeOdds :
                        selectedOutcome === 'draw' ? selectedMatch.drawOdds :
                        selectedMatch.awayOdds,
                        betAmount
                      ) - betAmount).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">기대값:</span>
                    <span className={`font-medium ${
                      calculateExpectedValue(
                        selectedOutcome === 'home' ? selectedMatch.homeWinProb :
                        selectedOutcome === 'draw' ? selectedMatch.drawProb :
                        selectedMatch.awayWinProb,
                        selectedOutcome === 'home' ? selectedMatch.homeOdds :
                        selectedOutcome === 'draw' ? selectedMatch.drawOdds :
                        selectedMatch.awayOdds,
                        betAmount
                      ) >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      ₩{calculateExpectedValue(
                        selectedOutcome === 'home' ? selectedMatch.homeWinProb :
                        selectedOutcome === 'draw' ? selectedMatch.drawProb :
                        selectedMatch.awayWinProb,
                        selectedOutcome === 'home' ? selectedMatch.homeOdds :
                        selectedOutcome === 'draw' ? selectedMatch.drawOdds :
                        selectedMatch.awayOdds,
                        betAmount
                      ).toLocaleString()}
                    </span>
                  </div>
                </div>
              )}

              <button
                disabled={!selectedOutcome || betAmount < 1000}
                className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
              >
                베팅하기
              </button>
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <Calculator className="w-12 h-12 mx-auto mb-3 text-gray-400" />
              <p>베팅할 경기를 선택해주세요</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

