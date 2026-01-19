"""
워크플로우 매니저
에이전트들의 실행 순서와 데이터 흐름을 관리
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class WorkflowManager:
    """워크플로우 실행 관리자"""

    def __init__(self):
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []

    def register_workflow(
        self,
        name: str,
        agents: List[BaseAgent],
        execution_type: str = "sequential",
        description: str = ""
    ):
        """워크플로우 등록"""
        self.workflows[name] = {
            "agents": agents,
            "execution_type": execution_type,  # "sequential", "parallel", "conditional"
            "description": description,
            "created_at": datetime.now(),
            "execution_count": 0
        }

        logger.info(f"워크플로우 '{name}' 등록 완료 ({len(agents)}개 에이전트)")

    async def execute_workflow(
        self,
        name: str,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """워크플로우 실행"""
        if name not in self.workflows:
            raise ValueError(f"워크플로우 '{name}'을 찾을 수 없습니다")

        workflow = self.workflows[name]
        context = context or {}

        start_time = datetime.now()
        execution_id = f"{name}_{start_time.strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"워크플로우 '{name}' 실행 시작: {task[:50]}...")

        try:
            # 실행 타입에 따른 분기
            if workflow["execution_type"] == "sequential":
                results = await self._execute_sequential(workflow["agents"], task, context)
            elif workflow["execution_type"] == "parallel":
                results = await self._execute_parallel(workflow["agents"], task, context)
            elif workflow["execution_type"] == "conditional":
                results = await self._execute_conditional(workflow["agents"], task, context)
            else:
                raise ValueError(f"지원하지 않는 실행 타입: {workflow['execution_type']}")

            # 실행 통계 업데이트
            workflow["execution_count"] += 1
            execution_time = (datetime.now() - start_time).total_seconds()

            # 실행 기록 저장
            execution_record = {
                "execution_id": execution_id,
                "workflow_name": name,
                "task": task,
                "start_time": start_time,
                "execution_time": execution_time,
                "agent_count": len(workflow["agents"]),
                "status": "completed",
                "results_summary": self._summarize_results(results)
            }
            self.execution_history.append(execution_record)

            logger.info(f"워크플로우 '{name}' 실행 완료 ({execution_time:.2f}초)")

            return {
                "execution_id": execution_id,
                "workflow_name": name,
                "status": "completed",
                "execution_time": execution_time,
                "agent_results": results,
                "final_context": context,
                "summary": self._generate_execution_summary(name, results, execution_time)
            }

        except Exception as e:
            logger.error(f"워크플로우 '{name}' 실행 오류: {e}")

            # 오류 기록
            execution_record = {
                "execution_id": execution_id,
                "workflow_name": name,
                "task": task,
                "start_time": start_time,
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "status": "error",
                "error": str(e)
            }
            self.execution_history.append(execution_record)

            raise

    async def _execute_sequential(
        self,
        agents: List[BaseAgent],
        task: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """순차 실행"""
        results = []

        for i, agent in enumerate(agents):
            logger.info(f"에이전트 {i+1}/{len(agents)} 실행: {agent.name}")

            # 에이전트 실행
            result = await agent.run(task, context)
            results.append(result)

            # 다음 에이전트를 위한 컨텍스트 업데이트
            context.update(result)

            # 오류 발생 시 중단
            if result.get("status") == "error":
                logger.warning(f"에이전트 '{agent.name}' 오류로 워크플로우 중단")
                break

        return results

    async def _execute_parallel(
        self,
        agents: List[BaseAgent],
        task: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """병렬 실행"""
        logger.info(f"{len(agents)}개 에이전트 병렬 실행")

        # 모든 에이전트를 동시에 실행
        tasks = [agent.run(task, context.copy()) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "agent": agents[i].name,
                    "status": "error",
                    "error": str(result)
                })
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_conditional(
        self,
        agents: List[BaseAgent],
        task: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """조건부 실행 (현재는 순차 실행과 동일, 향후 확장)"""
        # TODO: 조건부 로직 구현
        return await self._execute_sequential(agents, task, context)

    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """결과 요약"""
        total_agents = len(results)
        successful_agents = len([r for r in results if r.get("status") != "error"])
        failed_agents = total_agents - successful_agents

        return {
            "total_agents": total_agents,
            "successful": successful_agents,
            "failed": failed_agents,
            "success_rate": successful_agents / total_agents if total_agents > 0 else 0
        }

    def _generate_execution_summary(
        self,
        workflow_name: str,
        results: List[Dict[str, Any]],
        execution_time: float
    ) -> str:
        """실행 요약 생성"""
        summary = self._summarize_results(results)

        return (
            f"워크플로우 '{workflow_name}' 완료: "
            f"{summary['successful']}/{summary['total_agents']} 에이전트 성공 "
            f"({execution_time:.2f}초)"
        )

    def get_workflow_info(self, name: str) -> Optional[Dict[str, Any]]:
        """워크플로우 정보 조회"""
        if name not in self.workflows:
            return None

        workflow = self.workflows[name]
        return {
            "name": name,
            "description": workflow["description"],
            "execution_type": workflow["execution_type"],
            "agent_count": len(workflow["agents"]),
            "agents": [agent.name for agent in workflow["agents"]],
            "created_at": workflow["created_at"].isoformat(),
            "execution_count": workflow["execution_count"]
        }

    def list_workflows(self) -> List[Dict[str, Any]]:
        """등록된 워크플로우 목록"""
        return [self.get_workflow_info(name) for name in self.workflows.keys()]

    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """실행 기록 조회"""
        return sorted(
            self.execution_history,
            key=lambda x: x["start_time"],
            reverse=True
        )[:limit]
