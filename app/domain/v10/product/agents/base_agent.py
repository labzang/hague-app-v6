"""
베이스 에이전트 클래스
모든 에이전트가 상속받는 기본 클래스
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """모든 에이전트의 베이스 클래스"""

    def __init__(
        self,
        name: str,
        instruction: str,
        server_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.instruction = instruction
        self.server_names = server_names or []
        self.metadata = metadata or {}
        self.execution_count = 0
        self.last_execution = None

        logger.info(f"에이전트 '{name}' 초기화 완료")

    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트 실행 로직 (하위 클래스에서 구현)

        Args:
            task: 실행할 작업 설명
            context: 실행 컨텍스트 (이전 에이전트 결과 등)

        Returns:
            실행 결과 딕셔너리
        """
        pass

    async def run(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트 실행 래퍼 (공통 로직 처리)
        """
        start_time = datetime.now()

        try:
            logger.info(f"에이전트 '{self.name}' 실행 시작: {task[:50]}...")

            # 실행 전 검증
            await self._pre_execute_validation(task, context)

            # 메인 실행 로직
            result = await self.execute(task, context)

            # 실행 후 처리
            result = await self._post_execute_processing(result, start_time)

            # 통계 업데이트
            self.execution_count += 1
            self.last_execution = datetime.now()

            logger.info(f"에이전트 '{self.name}' 실행 완료")
            return result

        except Exception as e:
            logger.error(f"에이전트 '{self.name}' 실행 오류: {e}")

            # 오류 결과 반환
            return {
                "agent": self.name,
                "status": "error",
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }

    async def _pre_execute_validation(self, task: str, context: Dict[str, Any]):
        """실행 전 검증"""
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")

        if not isinstance(context, dict):
            raise ValueError("Context must be a dictionary")

    async def _post_execute_processing(self, result: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
        """실행 후 공통 처리"""
        if not isinstance(result, dict):
            result = {"result": result}

        # 메타데이터 추가
        result.update({
            "agent": self.name,
            "status": result.get("status", "success"),
            "execution_time": (datetime.now() - start_time).total_seconds(),
            "timestamp": datetime.now().isoformat(),
            "execution_count": self.execution_count + 1
        })

        return result

    def get_info(self) -> Dict[str, Any]:
        """에이전트 정보 반환"""
        return {
            "name": self.name,
            "instruction": self.instruction,
            "server_names": self.server_names,
            "metadata": self.metadata,
            "execution_count": self.execution_count,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', executions={self.execution_count})>"


class MockAgent(BaseAgent):
    """테스트용 모킹 에이전트"""

    def __init__(self, name: str = "mock_agent", response: str = "Mock response"):
        super().__init__(
            name=name,
            instruction="Mock agent for testing purposes",
            server_names=[]
        )
        self.response = response

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """모킹 실행"""
        await asyncio.sleep(0.1)  # 실행 시간 시뮬레이션

        return {
            "result": self.response,
            "task": task,
            "context_keys": list(context.keys()),
            "mock": True
        }
