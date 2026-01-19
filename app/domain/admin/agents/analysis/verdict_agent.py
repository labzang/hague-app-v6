"""
판독 에이전트
기존 EXAONE 기반 판독 시스템을 에이전트로 래핑
"""

from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.agents.analysis import get_exaone_analysis_agent


class VerdictAgent(BaseAgent):
    """EXAONE 기반 상세 판독 에이전트"""

    def __init__(self):
        super().__init__(
            name="verdict_agent",
            instruction="""You are a detailed email analysis agent using EXAONE model.
            Provide thorough analysis of suspicious or uncertain emails.

            Your capabilities:
            1. Deep contextual analysis of email content
            2. Sophisticated reasoning about spam patterns
            3. Confidence adjustment based on detailed examination
            4. Explanation of decision rationale
            """,
            server_names=["filesystem"],  # EXAONE 모델 접근용
            metadata={
                "model": "EXAONE-2.4B",
                "domain": "Email Content Analysis",
                "specialization": "Uncertain Cases"
            }
        )
        self.analysis_agent = None

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """상세 판독 실행"""
        # 이메일 데이터 및 이전 분석 결과 추출
        email_data = context.get("email", {})
        koelectra_result = context.get("koelectra_result", {})

        if not email_data:
            raise ValueError("Email data not found in context")

        subject = email_data.get("subject", "")
        content = email_data.get("content", "")

        # MCP 래퍼 초기화 (지연 로딩)
        if self.analysis_agent is None:
            self.analysis_agent = get_exaone_analysis_agent()

        # EXAONE 기반 상세 분석
        analysis_result = await self.analysis_agent.analyze_email(
            email_subject=subject,
            email_content=content,
            koelectra_result=koelectra_result,
            analysis_type="detailed"
        )

        # 결과 구성
        verdict = analysis_result.get("verdict", "uncertain")
        confidence_adjustment = analysis_result.get("confidence_adjustment", 0.0)
        exaone_response = analysis_result.get("exaone_response", "")

        # 최종 신뢰도 계산
        base_confidence = koelectra_result.get("confidence", 0.5)
        final_confidence = min(0.99, base_confidence + confidence_adjustment)

        return {
            "verdict": verdict,
            "confidence_adjustment": confidence_adjustment,
            "final_confidence": final_confidence,
            "analysis": exaone_response,
            "reasoning": self._extract_reasoning(exaone_response),
            "exaone_result": analysis_result,
            "recommendation": self._get_final_recommendation(verdict, final_confidence)
        }

    def _extract_reasoning(self, exaone_response: str) -> str:
        """EXAONE 응답에서 추론 과정 추출"""
        # 간단한 키워드 기반 추론 추출
        if "스팸" in exaone_response or "차단" in exaone_response:
            return "스팸 패턴 감지됨"
        elif "정상" in exaone_response or "안전" in exaone_response:
            return "정상 메일로 판단됨"
        elif "불확실" in exaone_response or "애매" in exaone_response:
            return "판단이 어려운 경계 사례"
        else:
            return "상세 분석 완료"

    def _get_final_recommendation(self, verdict: str, confidence: float) -> str:
        """최종 권장사항"""
        if verdict == "spam" and confidence > 0.8:
            return "block_immediately"
        elif verdict == "normal" and confidence > 0.8:
            return "allow_immediately"
        else:
            return "manual_review_recommended"
