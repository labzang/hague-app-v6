"""
EXAONE 기반 이메일 분석 에이전트
EXAONE 툴들을 활용한 이메일 스팸 분석 및 판정을 수행하는 에이전트
"""

import logging
from typing import Dict, Any, List

from app.agents.base_agent import BaseAgent
from app.tools.analysis.verdict_tools import exaone_tools
from app.tools.executors.tool_executor import SimpleToolExecutor

logger = logging.getLogger(__name__)


class ExaoneAnalysisAgent(BaseAgent):
    """EXAONE 기반 이메일 분석 에이전트"""

    def __init__(self):
        super().__init__(
            name="ExaoneAnalysisAgent",
            instruction="EXAONE 모델을 사용하여 이메일의 스팸 여부를 정밀 분석합니다.",
            metadata={
                "model": "EXAONE",
                "capabilities": ["email_analysis", "spam_detection", "detailed_analysis", "quick_verdict"],
                "tools": ["exaone_spam_analyzer", "exaone_quick_verdict", "exaone_detailed_analyzer"]
            }
        )
        self.tools = exaone_tools
        self.tool_executor = SimpleToolExecutor(exaone_tools)
        logger.info("EXAONE 분석 에이전트 초기화 완료")

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트 실행 로직

        Args:
            task: 실행할 작업 ("analyze_email")
            context: 실행 컨텍스트 (이메일 정보 및 KoELECTRA 결과)

        Returns:
            분석 결과 딕셔너리
        """
        if task == "analyze_email":
            return await self.analyze_email(
                email_subject=context.get("email_subject", ""),
                email_content=context.get("email_content", ""),
                koelectra_result=context.get("koelectra_result", {}),
                analysis_type=context.get("analysis_type", "detailed")
            )
        else:
            raise ValueError(f"지원하지 않는 작업: {task}")

    async def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        특정 툴 실행

        Args:
            tool_name: 실행할 툴 이름
            **kwargs: 툴에 전달할 인자들

        Returns:
            툴 실행 결과

        Raises:
            ValueError: 툴을 찾을 수 없는 경우
        """
        try:
            # 툴 이름으로 툴 찾기
            tool_map = {tool.name: tool for tool in self.tools}

            if tool_name not in tool_map:
                available_tools = list(tool_map.keys())
                raise ValueError(f"툴 '{tool_name}'을 찾을 수 없습니다. 사용 가능한 툴: {available_tools}")

            selected_tool = tool_map[tool_name]
            result = await selected_tool.ainvoke(kwargs)

            logger.info(f"툴 '{tool_name}' 실행 완료")
            return result

        except Exception as e:
            logger.error(f"툴 '{tool_name}' 실행 오류: {e}")
            raise

    async def analyze_email(
        self,
        email_subject: str,
        email_content: str,
        koelectra_result: Dict[str, Any],
        analysis_type: str = "detailed"
    ) -> Dict[str, Any]:
        """
        EXAONE을 사용한 이메일 분석

        Args:
            email_subject: 이메일 제목
            email_content: 이메일 내용
            koelectra_result: KoELECTRA 분석 결과
            analysis_type: 분석 타입 ("detailed" 또는 "quick")

        Returns:
            분석 결과 딕셔너리
        """
        try:
            logger.info(f"EXAONE 이메일 분석 시작: {analysis_type} 타입")

            if analysis_type == "detailed":
                response = await self.execute_tool(
                    "exaone_detailed_analyzer",
                    email_subject=email_subject,
                    email_content=email_content,
                    koelectra_result=koelectra_result
                )
            else:
                email_text = f"{email_subject} {email_content}"
                confidence = koelectra_result.get("confidence", 0.0)
                response = await self.execute_tool(
                    "exaone_quick_verdict",
                    email_text=email_text,
                    koelectra_confidence=confidence
                )

            # 응답 분석
            response_lower = response.lower()
            if "스팸" in response_lower or "차단" in response_lower:
                verdict = "spam"
                confidence_adjustment = 0.1
            elif "정상" in response_lower or "안전" in response_lower:
                verdict = "normal"
                confidence_adjustment = 0.1
            elif "불확실" in response_lower or "보류" in response_lower:
                verdict = "uncertain"
                confidence_adjustment = 0.0
            else:
                # KoELECTRA 결과 따름
                verdict = "spam" if koelectra_result.get("is_spam") else "normal"
                confidence_adjustment = 0.05

            result = {
                "agent_name": self.name,
                "verdict": verdict,
                "confidence_adjustment": confidence_adjustment,
                "analysis_type": analysis_type,
                "exaone_response": response,
                "analysis_summary": f"EXAONE 분석: {verdict} (신뢰도 조정: +{confidence_adjustment:.2f})",
                "tool_used": "exaone_detailed_analyzer" if analysis_type == "detailed" else "exaone_quick_verdict",
                "execution_time": None  # BaseAgent에서 자동 계산됨
            }

            logger.info(f"EXAONE 이메일 분석 완료: {verdict}")
            return result

        except Exception as e:
            logger.error(f"EXAONE 이메일 분석 오류: {e}")
            raise

    def get_available_tools(self) -> List[str]:
        """사용 가능한 툴 목록 반환"""
        return [tool.name for tool in self.tools]

    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """특정 툴의 정보 반환"""
        tool_map = {tool.name: tool for tool in self.tools}

        if tool_name not in tool_map:
            return {"error": f"툴 '{tool_name}'을 찾을 수 없습니다"}

        tool = tool_map[tool_name]
        return {
            "name": tool.name,
            "description": tool.description,
            "args_schema": tool.args_schema.schema() if tool.args_schema else None
        }

    def get_capabilities(self) -> List[str]:
        """에이전트 능력 목록 반환"""
        return self.metadata.get("capabilities", [])


# 전역 EXAONE 분석 에이전트 인스턴스
_exaone_analysis_agent: ExaoneAnalysisAgent = None

def get_exaone_analysis_agent() -> ExaoneAnalysisAgent:
    """EXAONE 분석 에이전트 싱글톤 인스턴스 가져오기"""
    global _exaone_analysis_agent
    if _exaone_analysis_agent is None:
        _exaone_analysis_agent = ExaoneAnalysisAgent()
        logger.info("새로운 EXAONE 분석 에이전트 인스턴스 생성")
    return _exaone_analysis_agent

# 호환성을 위한 별칭
ExaoneAnalysisService = ExaoneAnalysisAgent
get_exaone_analysis_service = get_exaone_analysis_agent

# MCP 호환성을 위한 별칭
MCPAgentWrapper = ExaoneAnalysisAgent

def get_mcp_agent_wrapper() -> ExaoneAnalysisAgent:
    """MCP 에이전트 래퍼 인스턴스 가져오기 (호환성 함수)"""
    return get_exaone_analysis_agent()


# 워크플로우 정보 조회
def get_workflow_info() -> Dict[str, Any]:
    """판독 에이전트 워크플로우 정보 반환"""
    return {
        "agent_name": "EXAONE Verdict Agent",
        "description": "EXAONE 기반 이메일 스팸 정밀 판독 에이전트",
        "workflow_steps": [
            "Initialize Analysis",
            "Generate Prompt",
            "EXAONE Analysis",
            "Verdict Decision",
            "Finalize Verdict"
        ],
        "analysis_types": {
            "detailed": "상세 분석 (신뢰도 ≤ 0.8)",
            "quick": "빠른 분석 (신뢰도 > 0.8)"
        },
        "verdict_options": ["spam", "normal", "uncertain"],
        "features": [
            "적응적 분석 타입 선택",
            "신뢰도 기반 프롬프트 생성",
            "상세 스팸 분석",
            "신뢰도 조정 기능"
        ]
    }


# MCP 라우터용 간편 인터페이스들
async def analyze_email_with_tools(
    email_subject: str,
    email_content: str,
    koelectra_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    툴 기반 이메일 판독 분석 (MCP 라우터용)

    Args:
        email_subject: 이메일 제목
        email_content: 이메일 내용
        koelectra_result: KoELECTRA 분석 결과

    Returns:
        판독 결과 딕셔너리
    """
    analysis_agent = get_exaone_analysis_agent()

    # 분석 타입 결정
    confidence = koelectra_result.get("confidence", 0.0)
    analysis_type = "quick" if confidence > 0.8 else "detailed"

    # 툴 기반 분석 실행
    result = await analysis_agent.analyze_email(
        email_subject=email_subject,
        email_content=email_content,
        koelectra_result=koelectra_result,
        analysis_type=analysis_type
    )

    return result


async def analyze_email_verdict(
    email_subject: str,
    email_content: str,
    koelectra_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    이메일 판독 분석 메인 함수 (워크플로우 기반 호환성)

    Args:
        email_subject: 이메일 제목
        email_content: 이메일 내용
        koelectra_result: KoELECTRA 분석 결과

    Returns:
        판독 결과 딕셔너리
    """
    # 실제로는 동일한 에이전트 사용 (워크플로우 대신)
    return await analyze_email_with_tools(email_subject, email_content, koelectra_result)


async def quick_verdict(
    email_text: str,
    koelectra_confidence: float
) -> Dict[str, Any]:
    """빠른 판정 (고신뢰도 케이스용)"""
    from app.tools.analysis.verdict_tools import exaone_quick_verdict

    try:
        response = await exaone_quick_verdict.ainvoke({
            "email_text": email_text,
            "koelectra_confidence": koelectra_confidence
        })

        # 간단한 판정 분석
        response_lower = response.lower()
        if "스팸" in response_lower:
            verdict = "spam"
        elif "정상" in response_lower:
            verdict = "normal"
        else:
            verdict = "uncertain"

        result = {
            "verdict": verdict,
            "confidence_adjustment": 0.05,
            "analysis_type": "quick",
            "analysis_summary": f"빠른 판정: {verdict}",
            "exaone_response": response,
            "processing_steps": ["quick_verdict_completed"]
        }

        return result

    except Exception as e:
        logger.error(f"EXAONE 빠른 판정 실패: {e}")
        raise
