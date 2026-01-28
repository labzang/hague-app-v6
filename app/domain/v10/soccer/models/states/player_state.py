"""선수 데이터 처리 상태 스키마."""
from typing import List, Dict, Any

from app.domain.v10.soccer.models.states.base_state import BaseProcessingState


class PlayerProcessingState(BaseProcessingState):
    """선수 데이터 처리 상태 스키마.

    LangGraph StateGraph에서 사용하는 상태 정의.
    각 노드에서 이 상태를 읽고 업데이트합니다.
    """

    # 입력 데이터 (BaseProcessingState에서 상속)
    items: List[Dict[str, Any]]

    # 검증 결과 (BaseProcessingState에서 상속)
    validation_errors: List[Dict[str, Any]]

    # 전략 판단 결과 (BaseProcessingState에서 상속)
    strategy_type: str  # "policy" | "rule"

    # 정규화된 데이터 (Rule 기반 처리용)
    normalized_items: List[Dict[str, Any]]

    # 정책 기반 처리 결과
    policy_result: Dict[str, Any]

    # 규칙 기반 처리 결과
    rule_result: Dict[str, Any]

    # 데이터베이스 저장 결과
    db_result: Dict[str, Any]

    # 최종 결과 (BaseProcessingState에서 상속)
    final_result: Dict[str, Any]

