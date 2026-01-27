"""
팩트 체킹 에이전트
제시된 연구 오케스트레이터의 팩트 체커를 구현
"""

from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent


class FactCheckerAgent(BaseAgent):
    """팩트 체킹 전문 에이전트"""

    def __init__(self):
        super().__init__(
            name="fact_checker",
            instruction="""You are a meticulous fact checker. Your role is to:
            1. Verify claims by cross-referencing sources
            2. Check dates, statistics, and technical details for accuracy
            3. Identify any contradictions or inconsistencies
            4. Rate the reliability of information sources
            """,
            server_names=["brave", "fetch", "filesystem"],
            metadata={
                "specialization": "Fact Verification",
                "focus": "Accuracy and Consistency",
                "verification_methods": ["cross_reference", "source_validation", "consistency_check"]
            }
        )

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """팩트 체킹 실행"""
        # 이전 검색 결과에서 소스들 가져오기
        sources = context.get("sources", [])
        search_query = context.get("search_query", task)

        if not sources:
            return {
                "status": "no_sources",
                "message": "No sources provided for fact checking"
            }

        # 각 소스에 대한 팩트 체킹 수행
        fact_check_results = []

        for source in sources:
            fact_check = await self._verify_source(source, search_query)
            fact_check_results.append(fact_check)

        # 전체 신뢰도 계산
        overall_reliability = self._calculate_overall_reliability(fact_check_results)

        # 모순점 탐지
        contradictions = self._detect_contradictions(fact_check_results)

        return {
            "fact_check_results": fact_check_results,
            "overall_reliability": overall_reliability,
            "contradictions": contradictions,
            "verification_summary": self._generate_verification_summary(fact_check_results),
            "recommendations": self._generate_recommendations(overall_reliability, contradictions)
        }

    async def _verify_source(self, source: Dict[str, Any], query: str) -> Dict[str, Any]:
        """개별 소스 검증"""
        # TODO: 실제 MCP 서버를 통한 검증 로직
        # 현재는 모킹 구현

        reliability_score = source.get("relevance_score", 0.5)
        source_type = source.get("source_type", "unknown")

        # 소스 타입별 신뢰도 조정
        type_multipliers = {
            "academic": 1.0,
            "technical": 0.9,
            "news": 0.8,
            "blog": 0.6,
            "unknown": 0.5
        }

        adjusted_reliability = reliability_score * type_multipliers.get(source_type, 0.5)

        return {
            "source_url": source.get("url", ""),
            "source_title": source.get("title", ""),
            "reliability_score": adjusted_reliability,
            "verification_status": "verified" if adjusted_reliability > 0.7 else "questionable",
            "issues_found": [] if adjusted_reliability > 0.8 else ["Low source reliability"],
            "verification_notes": f"Source type: {source_type}, Adjusted reliability: {adjusted_reliability:.2f}"
        }

    def _calculate_overall_reliability(self, results: List[Dict[str, Any]]) -> float:
        """전체 신뢰도 계산"""
        if not results:
            return 0.0

        scores = [r.get("reliability_score", 0.0) for r in results]
        return sum(scores) / len(scores)

    def _detect_contradictions(self, results: List[Dict[str, Any]]) -> List[str]:
        """모순점 탐지"""
        contradictions = []

        # 간단한 모순 탐지 로직
        verified_count = len([r for r in results if r.get("verification_status") == "verified"])
        questionable_count = len([r for r in results if r.get("verification_status") == "questionable"])

        if questionable_count > verified_count:
            contradictions.append("More questionable sources than verified ones")

        return contradictions

    def _generate_verification_summary(self, results: List[Dict[str, Any]]) -> str:
        """검증 요약 생성"""
        total = len(results)
        verified = len([r for r in results if r.get("verification_status") == "verified"])

        return f"Verified {verified} out of {total} sources. Overall reliability assessment completed."

    def _generate_recommendations(self, reliability: float, contradictions: List[str]) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        if reliability < 0.6:
            recommendations.append("Seek additional high-quality sources")

        if contradictions:
            recommendations.append("Resolve identified contradictions before proceeding")

        if reliability > 0.8:
            recommendations.append("Sources appear reliable, proceed with report generation")

        return recommendations
