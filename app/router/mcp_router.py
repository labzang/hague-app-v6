"""
KoELECTRA 게이트웨이 라우터 - 게이트웨이 및 상태 관리 기능
이메일 입력 → KoELECTRA 판별 → 조건 분기 → 판독 에이전트 호출 or 즉시 응답
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from datetime import datetime

# 로컬 imports
from app.service.verdict_agent import (
    analyze_email_verdict,
    analyze_email_with_tools,
    quick_verdict,
    get_mcp_agent_wrapper,
    EmailInput,
    GatewayResponse
)
from app.controller.mcp_controller import get_mcp_controller

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["Multi-Agent Communication Protocol"])

# 모델들은 verdict_agent 패키지로 이동됨

# API 엔드포인트들
@router.post("/analyze-email", response_model=GatewayResponse)
async def analyze_email(email: EmailInput):
    """
    이메일 스팸 분석 메인 엔드포인트
    KoELECTRA 게이트웨이 → 조건 분기 → 판독 에이전트 호출
    """
    try:
        logger.info(f"이메일 분석 시작: {email.subject[:50]}...")

        # 컨트롤러 인스턴스 가져오기
        controller = get_mcp_controller()

        # 세션 생성
        session_id = controller.create_session(email)
        session = controller.get_session(session_id)

        # 1. KoELECTRA 게이트웨이 분석
        koelectra_result = await controller.koelectra_gateway_analysis(email)
        controller.update_session(session_id, {
            "koelectra_result": koelectra_result,
            "confidence_score": koelectra_result["confidence"]
        })
        session.processing_steps.append("koelectra_completed")

        # 2. 라우팅 결정
        routing_decision = controller.determine_routing(koelectra_result)
        session.processing_steps.append(f"routed_to_{routing_decision}")

        # 3. 조건부 판독 에이전트 호출 (툴 기반)
        verdict_result = None
        if routing_decision == "verdict_agent":
            logger.info("판독 에이전트 호출 (툴 기반)")
            try:
                # 툴 기반 분석 사용
                verdict_result = await analyze_email_with_tools(
                    email.subject,
                    email.content,
                    koelectra_result
                )
                controller.update_session(session_id, {"verdict_result": verdict_result})
                session.processing_steps.append("tool_based_analysis_completed")
            except Exception as e:
                logger.warning(f"툴 기반 분석 실패, 기존 방식으로 대체: {e}")
                # 기존 워크플로우 방식으로 대체
                verdict_result = await analyze_email_verdict(
                    email.subject,
                    email.content,
                    koelectra_result
                )
                controller.update_session(session_id, {"verdict_result": verdict_result})
                session.processing_steps.extend(verdict_result.get("processing_steps", []))

        # 4. 최종 결정
        final_is_spam, final_confidence = controller.make_final_decision(
            koelectra_result, verdict_result, routing_decision
        )

        # 5. 세션 완료
        controller.update_session(session_id, {
            "status": "completed",
            "end_time": datetime.now(),
            "final_decision": routing_decision
        })

        # 응답 구성
        response = GatewayResponse(
            is_spam=final_is_spam,
            confidence=final_confidence,
            koelectra_decision=f"{'스팸' if koelectra_result['is_spam'] else '정상'} (신뢰도: {koelectra_result['confidence']:.3f})",
            exaone_analysis=verdict_result.get("exaone_response") if verdict_result else None,
            processing_path=" → ".join(session.processing_steps),
            timestamp=datetime.now(),
            metadata={
                "session_id": session_id,
                "koelectra_result": koelectra_result,
                "verdict_result": verdict_result,
                "routing_decision": routing_decision
            }
        )

        logger.info(f"분석 완료: 스팸={final_is_spam}, 라우팅={routing_decision}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"이메일 분석 중 오류: {e}")
        # 세션 오류 상태 업데이트
        if 'session_id' in locals():
            controller = get_mcp_controller()
            controller.update_session(session_id, {
                "status": "error",
                "error": str(e),
                "end_time": datetime.now()
            })
        raise HTTPException(status_code=500, detail=f"분석 처리 오류: {str(e)}")

@router.get("/sessions/{session_id}")
async def get_session_status(session_id: str):
    """세션 상태 조회"""
    controller = get_mcp_controller()
    session = controller.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

    processing_time = None
    if session.start_time and session.end_time:
        processing_time = (session.end_time - session.start_time).total_seconds()

    return {
        "session_id": session_id,
        "status": session.status,
        "processing_steps": session.processing_steps,
        "start_time": session.start_time,
        "end_time": session.end_time,
        "processing_time": processing_time,
        "koelectra_result": session.koelectra_result,
        "verdict_result": session.verdict_result,
        "error": session.error
    }

@router.get("/sessions")
async def list_sessions(limit: int = 50):
    """세션 목록 조회"""
    controller = get_mcp_controller()
    # 최근 세션들만 반환
    sorted_sessions = sorted(
        controller.processing_sessions.items(),
        key=lambda x: x[1].start_time,
        reverse=True
    )[:limit]

    return {
        "total_sessions": len(controller.processing_sessions),
        "returned_sessions": len(sorted_sessions),
        "sessions": [
            {
                "session_id": session_id,
                "status": session.status,
                "start_time": session.start_time,
                "email_subject": session.email_input.subject[:50] + "..." if len(session.email_input.subject) > 50 else session.email_input.subject,
                "final_decision": session.final_decision
            }
            for session_id, session in sorted_sessions
        ]
    }

@router.delete("/sessions/cleanup")
async def cleanup_sessions(max_age_hours: int = 24):
    """오래된 세션 정리"""
    controller = get_mcp_controller()
    cleaned_count = controller.cleanup_old_sessions(max_age_hours)
    return {
        "message": f"{cleaned_count}개의 오래된 세션이 정리되었습니다",
        "remaining_sessions": len(controller.processing_sessions),
        "max_age_hours": max_age_hours
    }

@router.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        controller = get_mcp_controller()
        # 서비스 상태 확인
        koelectra_status = "OK" if controller.spam_classifier else "Not Loaded"

        # 세션 통계
        session_stats = {
            "total": len(controller.processing_sessions),
            "processing": len([s for s in controller.processing_sessions.values() if s.status == "processing"]),
            "completed": len([s for s in controller.processing_sessions.values() if s.status == "completed"]),
            "error": len([s for s in controller.processing_sessions.values() if s.status == "error"])
        }

        return {
            "status": "healthy",
            "services": {
                "koelectra": koelectra_status,
                "verdict_agent": "Available"
            },
            "sessions": session_stats,
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"헬스 체크 실패: {str(e)}")

@router.get("/gateway-info")
async def get_gateway_info():
    """게이트웨이 정보 조회"""
    return {
        "gateway_type": "KoELECTRA Spam Detection Gateway",
        "components": {
            "gateway": {
                "name": "KoELECTRA Gateway",
                "model": "monologg/koelectra-small-v3-discriminator",
                "adapter": "LoRA Fine-tuned",
                "role": "Primary spam classification"
            },
            "verdict_agent": {
                "name": "EXAONE Verdict Agent",
                "model": "EXAONE-2.4B",
                "role": "Detailed analysis for uncertain cases"
            }
        },
        "processing_flow": [
            "Email Input",
            "KoELECTRA Gateway Analysis",
            "Routing Decision",
            "Conditional Verdict Agent Call",
            "Final Decision"
        ],
        "routing_thresholds": {
            "immediate_pass": "> 95% confidence (normal)",
            "immediate_block": "> 95% confidence (spam)",
            "verdict_agent": "≤ 95% confidence (uncertain)"
        },
        "session_management": {
            "tracking": "UUID-based session tracking",
            "cleanup": "Automatic cleanup of old sessions",
            "monitoring": "Real-time session status monitoring"
        }
    }

@router.get("/stats")
async def get_gateway_stats():
    """게이트웨이 통계 조회"""
    try:
        controller = get_mcp_controller()
        # 세션 통계 계산
        total_sessions = len(controller.processing_sessions)

        if total_sessions == 0:
            return {
                "total_sessions": 0,
                "message": "아직 처리된 세션이 없습니다"
            }

        # 상태별 통계
        status_counts = {}
        routing_counts = {}

        for session in controller.processing_sessions.values():
            # 상태별 카운트
            status = session.status
            status_counts[status] = status_counts.get(status, 0) + 1

            # 라우팅별 카운트
            if session.final_decision:
                routing_counts[session.final_decision] = routing_counts.get(session.final_decision, 0) + 1

        # 평균 처리 시간 계산
        completed_sessions = [s for s in controller.processing_sessions.values() if s.status == "completed" and s.end_time]
        avg_processing_time = None
        if completed_sessions:
            total_time = sum((s.end_time - s.start_time).total_seconds() for s in completed_sessions)
            avg_processing_time = total_time / len(completed_sessions)

        return {
            "total_sessions": total_sessions,
            "status_distribution": status_counts,
            "routing_distribution": routing_counts,
            "average_processing_time": f"{avg_processing_time:.2f}초" if avg_processing_time else "N/A",
            "completed_sessions": len(completed_sessions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

@router.get("/tools")
async def get_available_tools():
    """사용 가능한 EXAONE 툴 목록 조회"""
    try:
        mcp_wrapper = get_mcp_agent_wrapper()
        tools = mcp_wrapper.get_available_tools()

        tool_info = []
        for tool_name in tools:
            info = mcp_wrapper.get_tool_info(tool_name)
            tool_info.append(info)

        return {
            "available_tools": tools,
            "tool_details": tool_info,
            "total_tools": len(tools)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"툴 정보 조회 실패: {str(e)}")

@router.post("/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, payload: Dict[str, Any]):
    """특정 툴 직접 실행"""
    try:
        mcp_wrapper = get_mcp_agent_wrapper()

        # 툴 실행
        result = await mcp_wrapper.execute_tool(tool_name, **payload)

        return {
            "tool_name": tool_name,
            "payload": payload,
            "result": result,
            "timestamp": datetime.now()
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"툴 실행 실패: {str(e)}")
