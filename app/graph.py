# graph.py - EXAONE-2.4B 모델 사용 버전
from typing import Protocol, TypedDict

from langgraph.graph import END, StateGraph

from app.core.llm.providers.exaone_local import ExaoneLocalLLM


class LLMProtocol(Protocol):
    """LLM 인터페이스 프로토콜"""

    def invoke(self, prompt: str) -> str: ...


# 시스템 규칙 - 한국어로 친근하게 답변
SYSTEM_RULE = (
    "당신은 도움이 되는 한국어 AI 어시스턴트입니다. "
    "사용자의 질문에 정확하고 친근하게 답변해주세요. "
    "답변은 간결하면서도 유용한 정보를 포함해야 합니다."
)


# 상태 정의 - EXAONE 그래프
class ExaoneState(TypedDict):
    user_text: str
    assistant_text: str


# 더미 모델 클래스 정의
class DummyLLM:
    def invoke(self, prompt: str) -> str:
        return "안녕하세요! EXAONE 모델이 준비되지 않아 더미 응답을 드립니다."


# LLM 초기화
print("[시작] EXAONE-2.4B 모델 초기화...")
try:
    llm: LLMProtocol = ExaoneLocalLLM()
    print("[완료] EXAONE-2.4B 모델 준비 완료!")
except Exception as e:
    print(f"[경고] EXAONE 모델 로드 실패: {e}")
    llm = DummyLLM()


def generate_text(prompt: str) -> str:
    """텍스트 생성 함수"""
    try:
        response = llm.invoke(prompt)
        return str(response)
    except Exception as e:
        print(f"[오류] 텍스트 생성 실패: {e}")
        return "죄송합니다. 응답 생성 중 오류가 발생했습니다."


def answer_node(state):
    """답변 노드 - EXAONE 모델로 질문에 답변"""
    user = state["user_text"]

    prompt = f"""{SYSTEM_RULE}

사용자 질문: {user}

답변:"""

    # EXAONE 모델로 답변 생성
    raw = generate_text(prompt)

    # 응답 정리
    text = raw.strip()

    # 불필요한 접두사 제거
    if text.startswith("답변:"):
        text = text[3:].strip()

    # 너무 긴 응답은 적절히 자르기 (최대 1000자)
    if len(text) > 1000:
        text = text[:1000] + "..."

    return {
        **state,
        "assistant_text": text,
    }


def build_graph():
    """EXAONE 그래프 빌드"""
    print("[빌드] EXAONE 그래프 구성 중...")

    g = StateGraph(ExaoneState)

    # 답변 노드 추가
    g.add_node("answer", answer_node)

    # 시작점 설정하고 바로 종료
    g.set_entry_point("answer")
    g.add_edge("answer", END)

    compiled = g.compile()
    print("[완료] EXAONE 그래프 빌드 성공!")
    return compiled


# 그래프 초기화
try:
    graph = build_graph()
except Exception as e:
    print(f"[오류] EXAONE 그래프 초기화 실패: {e}")
    graph = None


def run_once(user_text: str) -> str:
    """EXAONE 그래프 실행"""
    try:
        if graph is None:
            return "EXAONE 그래프가 준비되지 않았습니다!"

        if not user_text or not isinstance(user_text, str):
            return "질문을 입력해주세요."

        # 초기 상태
        init_state: ExaoneState = {
            "user_text": user_text,
            "assistant_text": "",
        }

        # 그래프 실행
        result = graph.invoke(init_state)

        # 결과 반환
        return result.get("assistant_text", "답변을 생성할 수 없습니다.")

    except Exception as e:
        print(f"[오류] EXAONE 그래프 실행 실패: {e}")
        return f"죄송합니다. 오류가 발생했습니다: {e}"
