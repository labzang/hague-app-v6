"""
웹 검색 에이전트
제시된 연구 오케스트레이터의 검색 에이전트를 구현
"""

from typing import Dict, Any
from app.agents.base_agent import BaseAgent


class SearcherAgent(BaseAgent):
    """웹 검색 전문 에이전트"""

    def __init__(self):
        super().__init__(
            name="searcher",
            instruction="""You are an expert web researcher. Your role is to:
            1. Search for relevant, authoritative sources on the given topic
            2. Visit the most promising URLs to gather detailed information
            3. Return a structured summary of your findings with source URLs

            Focus on high-quality sources like academic papers, respected tech publications,
            and official documentation.

            Save each individual source in the output/sources/ folder. We only need up to 10 sources max.
            """,
            server_names=["brave", "fetch", "filesystem"],
            metadata={
                "specialization": "Web Research",
                "max_sources": 10,
                "focus": "High-quality authoritative sources"
            }
        )

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """웹 검색 실행"""
        # 현재는 모킹 구현 (실제 MCP 통합 시 교체)
        search_query = context.get("search_query", task)

        # TODO: 실제 MCP brave search 서버 연동
        # 현재는 모킹 데이터 반환
        mock_sources = [
            {
                "title": f"Research on {search_query}",
                "url": f"https://example.com/research/{search_query.replace(' ', '-')}",
                "summary": f"Comprehensive analysis of {search_query} with detailed findings.",
                "relevance_score": 0.95,
                "source_type": "academic"
            },
            {
                "title": f"Technical Overview: {search_query}",
                "url": f"https://tech-journal.com/{search_query.replace(' ', '-')}",
                "summary": f"Technical deep-dive into {search_query} methodologies.",
                "relevance_score": 0.88,
                "source_type": "technical"
            }
        ]

        return {
            "search_query": search_query,
            "sources_found": len(mock_sources),
            "sources": mock_sources,
            "search_summary": f"Found {len(mock_sources)} high-quality sources on {search_query}",
            "next_steps": ["fact_check_sources", "synthesize_information"]
        }
