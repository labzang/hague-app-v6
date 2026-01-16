"""
MCP 컨트롤러 - KoELECTRA 게이트웨이 비즈니스 로직
라우터에서 분리된 비즈니스 로직을 담당하는 컨트롤러 클래스
"""

from typing import Dict, Any, Optional
import asyncio
import logging
from datetime import datetime
import uuid

from fastapi import HTTPException

# 로컬 imports
from app.service.spam_classifier.inference import SpamClassifier
from app.service.verdict_agent import (
    EmailInput,
    ProcessingSessionState
)

# 로깅 설정
logger = logging.getLogger(__name__)


class McpController:
    """MCP 게이트웨이 컨트롤러 클래스

    KoELECTRA 게이트웨이 및 세션 관리의 비즈니스 로직을 담당합니다.
    """

    def __init__(self):
        """컨트롤러 초기화"""
        self.spam_classifier: Optional[SpamClassifier] = None
        self.processing_sessions: Dict[str, ProcessingSessionState] = {}
        logger.info("MCP 컨트롤러 초기화 완료")

    def get_spam_classifier(self) -> SpamClassifier:
        """스팸 분류기 인스턴스 가져오기"""
        if self.spam_classifier is None:
            try:
                self.spam_classifier = SpamClassifier(
                    model_path="app/model/spam/lora/run_20260115_1313",
                    base_model="monologg/koelectra-small-v3-discriminator"
                )
                logger.info("KoELECTRA 스팸 분류기 로드 완료")
            except Exception as e:
                logger.error(f"KoELECTRA 로드 실패: {e}")
                raise HTTPException(status_code=500, detail=f"KoELECTRA 초기화 실패: {e}")
        return self.spam_classifier

    def create_session(self, email: EmailInput) -> str:
        """새로운 처리 세션 생성"""
        session_id = str(uuid.uuid4())
        session = ProcessingSessionState(
            session_id=session_id,
            email_input=email,
            start_time=datetime.now(),
            processing_steps=["session_created"]
        )
        self.processing_sessions[session_id] = session
        logger.info(f"새로운 세션 생성: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[ProcessingSessionState]:
        """세션 조회"""
        return self.processing_sessions.get(session_id)

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """세션 업데이트"""
        if session_id in self.processing_sessions:
            session = self.processing_sessions[session_id]
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            return True
        return False

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """오래된 세션 정리"""
        current_time = datetime.now()
        to_remove = []

        for session_id, session in self.processing_sessions.items():
            age = (current_time - session.start_time).total_seconds() / 3600
            if age > max_age_hours:
                to_remove.append(session_id)

        for session_id in to_remove:
            del self.processing_sessions[session_id]
            logger.info(f"오래된 세션 정리: {session_id}")

        return len(to_remove)

    async def koelectra_gateway_analysis(self, email: EmailInput) -> Dict[str, Any]:
        """KoELECTRA 게이트웨이 분석"""
        try:
            logger.info("KoELECTRA 게이트웨이 분석 시작")

            classifier = self.get_spam_classifier()

            # 이메일 텍스트 결합
            email_text = f"{email.subject} {email.content}".strip()

            # KoELECTRA 추론
            result = await asyncio.to_thread(classifier.predict, email_text)

            logger.info(f"KoELECTRA 결과: 스팸={result['is_spam']}, 신뢰도={result['confidence']:.3f}")
            return result

        except Exception as e:
            logger.error(f"KoELECTRA 게이트웨이 오류: {e}")
            raise

    def determine_routing(self, koelectra_result: Dict[str, Any]) -> str:
        """라우팅 결정 로직"""
        confidence = koelectra_result["confidence"]
        is_spam = koelectra_result["is_spam"]

        # 고신뢰도 정상 메일: 즉시 통과
        if not is_spam and confidence > 0.95:
            return "immediate_pass"

        # 고신뢰도 스팸: 즉시 차단
        elif is_spam and confidence > 0.95:
            return "immediate_block"

        # 중간 신뢰도: 판독 에이전트 호출
        else:
            return "verdict_agent"

    def make_final_decision(
        self,
        koelectra_result: Dict[str, Any],
        verdict_result: Optional[Dict[str, Any]],
        routing_decision: str
    ) -> tuple[bool, float]:
        """최종 결정 로직"""
        base_confidence = koelectra_result["confidence"]

        if routing_decision == "immediate_pass":
            return False, base_confidence
        elif routing_decision == "immediate_block":
            return True, base_confidence
        elif routing_decision == "verdict_agent" and verdict_result:
            # 판독 에이전트 결과 적용
            verdict = verdict_result.get("verdict", "uncertain")
            confidence_adjustment = verdict_result.get("confidence_adjustment", 0.0)

            if verdict == "spam":
                return True, min(0.99, base_confidence + confidence_adjustment)
            elif verdict == "normal":
                return False, min(0.99, base_confidence + confidence_adjustment)
            else:
                # 불확실한 경우 KoELECTRA 결과 사용
                return koelectra_result["is_spam"], base_confidence
        else:
            # 기본값: KoELECTRA 결과 사용
            return koelectra_result["is_spam"], base_confidence


# 전역 컨트롤러 인스턴스 (싱글톤 패턴)
_controller_instance: Optional[McpController] = None


def get_mcp_controller() -> McpController:
    """MCP 컨트롤러 인스턴스 가져오기 (싱글톤)"""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = McpController()
        logger.info("새로운 MCP 컨트롤러 인스턴스 생성")
    return _controller_instance

