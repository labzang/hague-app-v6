# graph.py - 베이비 스텝 버전
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.core.llm.providers.midm_local import create_midm_local_llm

# 시스템 규칙 - 아기처럼 짧게 답하기
SYSTEM_RULE = (
    "너는 아기처럼 짧게 인사만 한다. "
    "반드시 1문장으로만 답한다. "
    "설명, 질문, 조언, 목록을 하지 않는다."
)


# 상태 정의 - 베이비 스텝
class BabyState(TypedDict):
    user_text: str
    assistant_text: str


# LLM 초기화
print("[시작] 베이비 그래프용 Midm 모델 초기화...")
try:
    llm = create_midm_local_llm()
    print("[완료] 베이비 그래프 모델 준비 완료!")
except Exception as e:
    print(f"[경고] 모델 로드 실패: {e}")

    # 더미 모델
    class DummyLLM:
        def invoke(self, prompt: str) -> str:
            return "안녕! 나는 베이비 봇이야!"

    llm = DummyLLM()


def generate_text(prompt: str, max_new_tokens: int = 32) -> str:
    """텍스트 생성 함수"""
    try:
        response = llm.invoke(prompt)
        return str(response)
    except Exception as e:
        print(f"[오류] 텍스트 생성 실패: {e}")
        return "안녕! 베이비 봇이야!"


def greet_node(state):
    """인사 노드 - 아기처럼 짧게 답하기"""
    user = state["user_text"]

    prompt = f"""[SYSTEM]
{SYSTEM_RULE}

[USER]
{user}

[ASSISTANT]
"""

    # 생성 결과에서 "인사 1문장"만 남기고 싶으면 후처리를 약하게 추가
    raw = generate_text(prompt, max_new_tokens=32)

    # 간단 정리 - 가장 마지막 문장만 뽑기
    text = raw.strip().splitlines()[-1].strip()

    # 너무 길면 자르기
    if len(text) > 50:
        text = text[:50] + "..."

    return {
        **state,
        "assistant_text": text,
    }


def build_graph():
    """베이비 그래프 빌드 - 초간단 버전"""
    print("[빌드] 베이비 그래프 구성 중...")

    g = StateGraph(BabyState)

    # 인사 노드만 추가
    g.add_node("greet", greet_node)

    # 시작점 설정하고 바로 종료
    g.set_entry_point("greet")
    g.add_edge("greet", END)

    compiled = g.compile()
    print("[완료] 베이비 그래프 빌드 성공!")
    return compiled


# 그래프 초기화
try:
    graph = build_graph()
except Exception as e:
    print(f"[오류] 베이비 그래프 초기화 실패: {e}")
    graph = None


def run_once(user_text: str) -> str:
    """베이비 그래프 실행 - 심플 버전"""
    try:
        if graph is None:
            return "베이비 그래프가 준비되지 않았어요!"

        if not user_text or not isinstance(user_text, str):
            return "뭐라고 했어요?"

        # 초기 상태 - 베이비 스텝
        init_state: BabyState = {
            "user_text": user_text,
            "assistant_text": "",
        }

        # 그래프 실행
        result = graph.invoke(init_state)

        # 결과 반환
        return result.get("assistant_text", "베이비가 말을 못해요...")

    except Exception as e:
        print(f"[오류] 베이비 그래프 실행 실패: {e}")
        return f"베이비가 울고 있어요: {e}"
