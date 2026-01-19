"""
MCP 애플리케이션 래퍼
기존 mcp_agent.app.MCPApp을 프로젝트에 맞게 래핑
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class AgentPlatformApp:
    """에이전트 플랫폼 메인 애플리케이션"""

    def __init__(self, name: str = "agent_platform"):
        self.name = name
        self.agents: Dict[str, Any] = {}
        self.workflows: Dict[str, Any] = {}
        self.context: Optional[Any] = None
        self.logger = logger

        # MCP 서버 설정 (향후 확장)
        self.mcp_servers = {
            "filesystem": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-filesystem"]
            },
            "brave": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-brave-search"]
            },
            "fetch": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-fetch"]
            }
        }

        logger.info(f"에이전트 플랫폼 '{name}' 초기화 완료")

    @asynccontextmanager
    async def run(self):
        """플랫폼 실행 컨텍스트"""
        try:
            logger.info(f"에이전트 플랫폼 '{self.name}' 시작")

            # 플랫폼 초기화
            await self._initialize()

            # 컨텍스트 생성 (향후 MCP 통합 시 실제 구현)
            self.context = MockMCPContext(self.mcp_servers)

            yield self

        except Exception as e:
            logger.error(f"플랫폼 실행 중 오류: {e}")
            raise
        finally:
            logger.info(f"에이전트 플랫폼 '{self.name}' 종료")
            await self._cleanup()

    async def _initialize(self):
        """플랫폼 초기화"""
        # 향후 MCP 서버 연결, 에이전트 로딩 등
        pass

    async def _cleanup(self):
        """플랫폼 정리"""
        # 향후 리소스 정리
        pass

    def register_agent(self, agent):
        """에이전트 등록"""
        self.agents[agent.name] = agent
        logger.info(f"에이전트 '{agent.name}' 등록 완료")

    def register_workflow(self, workflow):
        """워크플로우 등록"""
        self.workflows[workflow.name] = workflow
        logger.info(f"워크플로우 '{workflow.name}' 등록 완료")

    def get_agent(self, name: str):
        """에이전트 조회"""
        return self.agents.get(name)

    def get_workflow(self, name: str):
        """워크플로우 조회"""
        return self.workflows.get(name)

    def list_agents(self) -> List[str]:
        """등록된 에이전트 목록"""
        return list(self.agents.keys())

    def list_workflows(self) -> List[str]:
        """등록된 워크플로우 목록"""
        return list(self.workflows.keys())


class MockMCPContext:
    """MCP 컨텍스트 모킹 (향후 실제 MCP 통합 시 교체)"""

    def __init__(self, servers: Dict[str, Any]):
        self.servers = servers
        self.config = MockConfig(servers)

    class MockConfig:
        def __init__(self, servers):
            self.mcp = MockMCP(servers)

    class MockMCP:
        def __init__(self, servers):
            self.servers = {name: MockServer(config) for name, config in servers.items()}

    class MockServer:
        def __init__(self, config):
            self.args = config.get("args", [])
            self.command = config.get("command", "")
