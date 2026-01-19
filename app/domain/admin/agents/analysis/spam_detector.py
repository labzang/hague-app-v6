"""
스팸 탐지 에이전트
기존 KoELECTRA 스팸 분류기를 에이전트로 래핑
"""

import asyncio
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.services.spam_classifier.inference import SpamClassifier


class SpamDetectorAgent(BaseAgent):
    """KoELECTRA 기반 스팸 탐지 에이전트"""

    def __init__(self):
        super().__init__(
            name="spam_detector",
            instruction="""You are an expert spam email detector using KoELECTRA model.
            Analyze emails and classify them as spam or legitimate with confidence scores.

            Your capabilities:
            1. Korean email spam detection with high accuracy
            2. Confidence scoring for classification results
            3. Fast inference using LoRA fine-tuned model
            """,
            server_names=["filesystem"],  # 모델 파일 접근용
            metadata={
                "model": "KoELECTRA + LoRA",
                "domain": "Korean Email Spam Detection",
                "accuracy": "95%+"
            }
        )
        self.classifier = None

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """스팸 탐지 실행"""
        # 이메일 데이터 추출
        email_data = context.get("email", {})
        if not email_data:
            raise ValueError("Email data not found in context")

        subject = email_data.get("subject", "")
        content = email_data.get("content", "")
        sender = email_data.get("sender", "")

        # 분류기 초기화 (지연 로딩)
        if self.classifier is None:
            self.classifier = SpamClassifier(
                model_path="app/models/spam/lora/run_20260115_1313",
                base_model="monologg/koelectra-small-v3-discriminator"
            )

        # 이메일 텍스트 결합
        email_text = f"{subject} {content}".strip()

        # KoELECTRA 추론 (비동기 실행)
        koelectra_result = await asyncio.to_thread(
            self.classifier.predict,
            email_text
        )

        # 결과 구성
        is_spam = koelectra_result["is_spam"]
        confidence = koelectra_result["confidence"]

        return {
            "classification": "spam" if is_spam else "legitimate",
            "confidence": confidence,
            "koelectra_result": koelectra_result,
            "email_info": {
                "subject": subject[:50] + "..." if len(subject) > 50 else subject,
                "content_length": len(content),
                "sender": sender
            },
            "routing_recommendation": self._get_routing_recommendation(confidence, is_spam)
        }

    def _get_routing_recommendation(self, confidence: float, is_spam: bool) -> str:
        """라우팅 권장사항 결정"""
        if confidence > 0.95:
            return "immediate_decision"  # 고신뢰도: 즉시 결정
        elif confidence > 0.7:
            return "quick_verification"  # 중간신뢰도: 빠른 검증
        else:
            return "detailed_analysis"  # 저신뢰도: 상세 분석 필요
