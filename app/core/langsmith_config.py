"""LangSmith 모니터링 설정.

LangGraph 및 LangChain 실행을 LangSmith로 추적합니다.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def get_langsmith_config() -> Optional[dict]:
    """LangSmith 설정을 반환합니다.

    환경 변수가 설정되어 있으면 LangSmith 추적을 활성화합니다.

    Returns:
        LangSmith config 딕셔너리 또는 None (비활성화 시)
    """
    # 환경 변수 확인
    api_key = os.getenv("LANGSMITH_API_KEY")
    tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes")
    project_name = os.getenv("LANGCHAIN_PROJECT", "soccer-data-processing")

    if not tracing_v2 or not api_key:
        logger.debug("[LangSmith] 추적 비활성화 (환경 변수 미설정)")
        return None

    logger.info(f"[LangSmith] 추적 활성화: 프로젝트={project_name}")

    return {
        "configurable": {
            "thread_id": "default",
        },
        "metadata": {
            "project": project_name,
            "source": "soccer-orchestrator",
        },
        "tags": ["langgraph", "soccer", "data-processing"],
    }


def is_langsmith_enabled() -> bool:
    """LangSmith 추적이 활성화되어 있는지 확인합니다.

    Returns:
        True if LangSmith is enabled, False otherwise
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes")
    return bool(api_key and tracing_v2)
