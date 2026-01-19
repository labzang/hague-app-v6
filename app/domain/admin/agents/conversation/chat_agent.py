"""
채팅 에이전트
기존 chat_service.py를 에이전트로 변환
"""

from typing import Dict, Any, List, Optional
from app.agents.base_agent import BaseAgent


class ChatAgent(BaseAgent):
    """QLoRA 기반 대화 에이전트"""

    def __init__(self):
        super().__init__(
            name="chat_agent",
            instruction="""You are a conversational AI agent with QLoRA capabilities.
            Your role is to:
            1. Engage in natural conversations with users
            2. Maintain conversation history and context
            3. Provide helpful and accurate responses
            4. Support Korean and English languages
            """,
            server_names=["filesystem"],  # 모델 파일 접근용
            metadata={
                "model_type": "QLoRA",
                "languages": ["Korean", "English"],
                "specialization": "Conversational AI"
            }
        )
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """대화 실행"""
        user_message = context.get("message", task)
        session_id = context.get("session_id", "default")

        # 대화 히스토리 관리
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "session_id": session_id
        })

        # 히스토리 길이 제한
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

        # TODO: 실제 QLoRA 모델 호출 로직
        # 현재는 모킹 응답
        response = await self._generate_response(user_message, self.conversation_history)

        # 응답을 히스토리에 추가
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "session_id": session_id
        })

        return {
            "response": response,
            "session_id": session_id,
            "conversation_turn": len(self.conversation_history) // 2,
            "context_length": len(self._format_conversation_context()),
            "model_info": "QLoRA-based conversational model"
        }

    async def _generate_response(self, message: str, history: List[Dict[str, str]]) -> str:
        """응답 생성 (실제 구현 시 QLoRA 모델 호출)"""
        # TODO: 실제 QLoRA 모델 통합
        # 현재는 간단한 규칙 기반 응답

        if "안녕" in message or "hello" in message.lower():
            return "안녕하세요! 무엇을 도와드릴까요?"
        elif "도움" in message or "help" in message.lower():
            return "저는 QLoRA 기반 대화 에이전트입니다. 다양한 질문에 답변해드릴 수 있어요."
        elif "고마워" in message or "thank" in message.lower():
            return "천만에요! 더 도움이 필요하시면 언제든 말씀해주세요."
        else:
            return f"'{message}'에 대해 말씀해주셔서 감사합니다. 더 자세히 설명해주시면 더 도움을 드릴 수 있어요."

    def _format_conversation_context(self) -> str:
        """대화 컨텍스트 포맷팅"""
        context = ""
        for entry in self.conversation_history[-6:]:  # 최근 3턴만
            role = "사용자" if entry["role"] == "user" else "어시스턴트"
            context += f"{role}: {entry['content']}\n"
        return context

    def get_conversation_summary(self, session_id: str = "default") -> Dict[str, Any]:
        """대화 요약 조회"""
        session_messages = [
            msg for msg in self.conversation_history
            if msg.get("session_id") == session_id
        ]

        return {
            "session_id": session_id,
            "total_messages": len(session_messages),
            "conversation_turns": len(session_messages) // 2,
            "recent_messages": session_messages[-4:] if session_messages else []
        }
