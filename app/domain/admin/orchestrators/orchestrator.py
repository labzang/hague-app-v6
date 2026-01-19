"""
메인 오케스트레이터
제시된 연구 오케스트레이터 코드를 기반으로 한 통합 오케스트레이터
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from app.orchestrator.mcp_app import AgentPlatformApp
from app.orchestrator.workflow_manager import WorkflowManager
from app.agents.base_agent import BaseAgent
from app.agents.research.searcher import SearcherAgent
from app.agents.research.fact_checker import FactCheckerAgent
from app.agents.research.report_writer import ReportWriterAgent
from app.agents.analysis.spam_detector import SpamDetectorAgent
from app.agents.analysis.verdict_agent import VerdictAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """메인 오케스트레이터 - 제시된 코드 패턴 구현"""

    def __init__(
        self,
        available_agents: Optional[List[BaseAgent]] = None,
        plan_type: str = "full",
        plan_output_path: Optional[Path] = None,
        max_iterations: int = 5
    ):
        self.available_agents = available_agents or []
        self.plan_type = plan_type
        self.plan_output_path = plan_output_path
        self.max_iterations = max_iterations

        # 플랫폼과 워크플로우 매니저 초기화
        self.platform = AgentPlatformApp("research_orchestrator")
        self.workflow_manager = WorkflowManager()

        # 에이전트 등록
        self._register_agents()

        # 기본 워크플로우 등록
        self._register_default_workflows()

        logger.info("오케스트레이터 초기화 완료")

    def _register_agents(self):
        """에이전트 등록"""
        for agent in self.available_agents:
            self.platform.register_agent(agent)

    def _register_default_workflows(self):
        """기본 워크플로우 등록"""
        # 연구 워크플로우 (제시된 코드 패턴)
        research_agents = [
            SearcherAgent(),
            FactCheckerAgent(),
            ReportWriterAgent()
        ]

        self.workflow_manager.register_workflow(
            name="research_report",
            agents=research_agents,
            execution_type="sequential",
            description="Web research and report generation workflow"
        )

        # 스팸 분석 워크플로우 (기존 시스템)
        spam_agents = [
            SpamDetectorAgent(),
            VerdictAgent()
        ]

        self.workflow_manager.register_workflow(
            name="spam_analysis",
            agents=spam_agents,
            execution_type="sequential",
            description="Email spam detection and analysis workflow"
        )

        logger.info("기본 워크플로우 등록 완료")

    async def generate_str(
        self,
        message: str,
        request_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        제시된 코드의 generate_str 메서드 구현
        연구 태스크를 실행하고 결과를 문자열로 반환
        """
        try:
            logger.info(f"연구 태스크 실행 시작: {message[:100]}...")

            # 플랫폼 실행
            async with self.platform.run() as app:
                # 연구 워크플로우 실행
                result = await self.workflow_manager.execute_workflow(
                    name="research_report",
                    task=message,
                    context={
                        "search_query": message,
                        "request_params": request_params or {}
                    }
                )

                # 실행 계획 저장 (옵션)
                if self.plan_output_path:
                    await self._save_execution_plan(result)

                # 결과를 문자열로 변환
                return self._format_result_as_string(result)

        except Exception as e:
            logger.error(f"연구 태스크 실행 오류: {e}")
            return f"연구 실행 중 오류 발생: {str(e)}"

    async def execute_workflow(
        self,
        workflow_name: str,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """워크플로우 실행 (일반적인 인터페이스)"""
        async with self.platform.run() as app:
            return await self.workflow_manager.execute_workflow(
                workflow_name, task, context or {}
            )

    async def _save_execution_plan(self, result: Dict[str, Any]):
        """실행 계획 저장"""
        try:
            if not self.plan_output_path:
                return

            # 디렉토리 생성
            self.plan_output_path.parent.mkdir(parents=True, exist_ok=True)

            # 실행 계획 내용 생성
            plan_content = self._generate_execution_plan_content(result)

            # 파일 저장
            with open(self.plan_output_path, 'w', encoding='utf-8') as f:
                f.write(plan_content)

            logger.info(f"실행 계획 저장 완료: {self.plan_output_path}")

        except Exception as e:
            logger.warning(f"실행 계획 저장 실패: {e}")

    def _generate_execution_plan_content(self, result: Dict[str, Any]) -> str:
        """실행 계획 내용 생성"""
        execution_id = result.get("execution_id", "unknown")
        workflow_name = result.get("workflow_name", "unknown")
        execution_time = result.get("execution_time", 0)

        content = f"""# 실행 계획 보고서

## 기본 정보
- **실행 ID**: {execution_id}
- **워크플로우**: {workflow_name}
- **실행 시간**: {execution_time:.2f}초
- **생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 에이전트 실행 결과
"""

        agent_results = result.get("agent_results", [])
        for i, agent_result in enumerate(agent_results, 1):
            agent_name = agent_result.get("agent", f"Agent_{i}")
            status = agent_result.get("status", "unknown")
            exec_time = agent_result.get("execution_time", 0)

            content += f"""
### {i}. {agent_name}
- **상태**: {status}
- **실행 시간**: {exec_time:.2f}초
- **결과 요약**: {self._summarize_agent_result(agent_result)}
"""

        return content

    def _summarize_agent_result(self, result: Dict[str, Any]) -> str:
        """에이전트 결과 요약"""
        if result.get("status") == "error":
            return f"오류: {result.get('error', 'Unknown error')}"

        # 에이전트별 특화 요약
        agent_name = result.get("agent", "")

        if "searcher" in agent_name:
            sources_count = result.get("sources_found", 0)
            return f"{sources_count}개 소스 발견"
        elif "fact_checker" in agent_name:
            reliability = result.get("overall_reliability", 0)
            return f"신뢰도: {reliability:.1%}"
        elif "report_writer" in agent_name:
            word_count = result.get("word_count", 0)
            return f"{word_count}단어 보고서 생성"
        elif "spam_detector" in agent_name:
            classification = result.get("classification", "unknown")
            confidence = result.get("confidence", 0)
            return f"분류: {classification} (신뢰도: {confidence:.1%})"
        else:
            return "실행 완료"

    def _format_result_as_string(self, result: Dict[str, Any]) -> str:
        """결과를 문자열로 포맷팅"""
        execution_id = result.get("execution_id", "")
        workflow_name = result.get("workflow_name", "")
        summary = result.get("summary", "")

        # 보고서 내용 추출 (report_writer 결과에서)
        agent_results = result.get("agent_results", [])
        report_content = ""

        for agent_result in agent_results:
            if agent_result.get("agent") == "report_writer":
                report_content = agent_result.get("report_content", "")
                break

        if report_content:
            return report_content
        else:
            return f"워크플로우 '{workflow_name}' 실행 완료: {summary}"

    def get_available_workflows(self) -> List[Dict[str, Any]]:
        """사용 가능한 워크플로우 목록"""
        return self.workflow_manager.list_workflows()

    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """실행 기록 조회"""
        return self.workflow_manager.get_execution_history(limit)


# 제시된 코드의 example_usage 함수를 위한 헬퍼
async def create_research_orchestrator() -> Orchestrator:
    """연구 오케스트레이터 생성 (제시된 코드 패턴)"""

    # 연구 에이전트들 생성
    search_agent = SearcherAgent()
    fact_checker = FactCheckerAgent()
    report_writer = ReportWriterAgent()

    # 오케스트레이터 생성
    orchestrator = Orchestrator(
        available_agents=[search_agent, fact_checker, report_writer],
        plan_type="full",
        plan_output_path=Path("output/execution_plan.md"),
        max_iterations=5
    )

    return orchestrator
