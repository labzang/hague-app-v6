"""
보고서 작성 에이전트
제시된 연구 오케스트레이터의 보고서 작성자를 구현
"""

from typing import Dict, Any
from datetime import datetime
from app.agents.base_agent import BaseAgent


class ReportWriterAgent(BaseAgent):
    """보고서 작성 전문 에이전트"""

    def __init__(self):
        super().__init__(
            name="report_writer",
            instruction="""You are a technical report writer specializing in research
            documents. Your role is to:
            1. Create well-structured, professional reports
            2. Include proper citations and references
            3. Balance technical depth with clarity
            4. Synthesize information from multiple sources

            Save your report to the filesystem with appropriate formatting.
            """,
            server_names=["filesystem", "fetch"],
            metadata={
                "specialization": "Technical Report Writing",
                "output_format": "Markdown",
                "focus": "Professional Documentation"
            }
        )

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """보고서 작성 실행"""
        # 이전 단계들의 결과 수집
        search_results = context.get("sources", [])
        fact_check_results = context.get("fact_check_results", [])
        search_query = context.get("search_query", task)
        overall_reliability = context.get("overall_reliability", 0.0)

        # 보고서 생성
        report_content = await self._generate_report(
            search_query, search_results, fact_check_results, overall_reliability
        )

        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{search_query.replace(' ', '_')}_{timestamp}.md"

        # TODO: 실제 파일 시스템에 저장 (MCP filesystem 서버 사용)
        # 현재는 메모리에만 보관

        return {
            "report_content": report_content,
            "filename": filename,
            "word_count": len(report_content.split()),
            "sections": self._count_sections(report_content),
            "sources_cited": len(search_results),
            "reliability_score": overall_reliability,
            "report_summary": self._generate_report_summary(search_query, len(search_results))
        }

    async def _generate_report(
        self,
        topic: str,
        sources: list,
        fact_checks: list,
        reliability: float
    ) -> str:
        """보고서 내용 생성"""

        report = f"""# Research Report: {topic}

## Executive Summary

This report presents a comprehensive analysis of {topic} based on {len(sources)} sources with an overall reliability score of {reliability:.2f}.

## Methodology

- **Search Strategy**: Comprehensive web search focusing on authoritative sources
- **Fact Checking**: Cross-reference verification of all sources
- **Quality Assessment**: Reliability scoring based on source credibility

## Key Findings

"""

        # 주요 발견사항 추가
        for i, source in enumerate(sources[:5], 1):
            report += f"### Finding {i}: {source.get('title', 'Untitled')}\n\n"
            report += f"{source.get('summary', 'No summary available')}\n\n"
            report += f"**Source**: [{source.get('title', 'Link')}]({source.get('url', '#')})\n\n"

        # 신뢰도 분석 섹션
        report += "## Source Reliability Analysis\n\n"

        verified_sources = [fc for fc in fact_checks if fc.get('verification_status') == 'verified']
        questionable_sources = [fc for fc in fact_checks if fc.get('verification_status') == 'questionable']

        report += f"- **Verified Sources**: {len(verified_sources)}\n"
        report += f"- **Questionable Sources**: {len(questionable_sources)}\n"
        report += f"- **Overall Reliability**: {reliability:.1%}\n\n"

        # 결론 섹션
        report += "## Conclusions\n\n"

        if reliability > 0.8:
            report += "The research findings are highly reliable based on authoritative sources.\n\n"
        elif reliability > 0.6:
            report += "The research findings are moderately reliable, with some sources requiring additional verification.\n\n"
        else:
            report += "The research findings require additional verification due to limited source reliability.\n\n"

        # 참고문헌
        report += "## References\n\n"
        for i, source in enumerate(sources, 1):
            report += f"{i}. [{source.get('title', 'Untitled')}]({source.get('url', '#')})\n"

        # 메타데이터
        report += f"\n---\n\n"
        report += f"**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Research Topic**: {topic}\n"
        report += f"**Sources Analyzed**: {len(sources)}\n"
        report += f"**Reliability Score**: {reliability:.2f}\n"

        return report

    def _count_sections(self, content: str) -> int:
        """섹션 수 계산"""
        return content.count('##')

    def _generate_report_summary(self, topic: str, source_count: int) -> str:
        """보고서 요약 생성"""
        return f"Generated comprehensive research report on '{topic}' analyzing {source_count} sources with structured findings and reliability assessment."
