"""FastAPI 기반 RAG 백엔드 서버 - 통합 버전

이 파일은 기존의 api_server.py와 main.py를 통합한 버전입니다.
- FastAPI 애플리케이션 설정
- 벡터스토어 및 RAG 체인 초기화
- API 엔드포인트 정의
- 로컬 Midm 모델 지원
"""

import os
import sys
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
import time

import uvicorn
import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# DB 테스트를 위한 설정 import
from app.core.config import settings

# 환경 변수 로드
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# 비동기 컨텍스트 매니저, 데이터베이스 초기화
def test_neon_db_connection() -> dict:
    """Neon DB 연결 테스트 및 상세 정보 반환."""
    logger.info("="*60)
    logger.info("[테스트] Neon DB 연결 테스트 시작")
    logger.info("="*60)

    # 설정 정보 출력
    logger.info(f"[설정] DATABASE_URL 환경변수 존재 여부: {settings.database_url_env is not None}")
    if settings.database_url_env:
        parsed_url = urlparse(settings.database_url_env)
        masked_url = f"{parsed_url.scheme}://{parsed_url.hostname}:{parsed_url.port}{parsed_url.path}"
        logger.info(f"[설정] DATABASE_URL (마스킹됨): {masked_url}")
    else:
        logger.info(f"[설정] POSTGRES_HOST: {settings.postgres_host}")
        logger.info(f"[설정] POSTGRES_PORT: {settings.postgres_port}")
        logger.info(f"[설정] POSTGRES_DB: {settings.postgres_db}")
        logger.info(f"[설정] POSTGRES_USER: {settings.postgres_user}")

    final_url = settings.database_url
    parsed_final = urlparse(final_url)
    masked_final = f"{parsed_final.scheme}://{parsed_final.hostname}:{parsed_final.port}{parsed_final.path}"
    logger.info(f"[설정] 최종 연결 문자열 (마스킹됨): {masked_final}")
    logger.info("-"*60)

    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            logger.info(f"[시도] 연결 시도 {retry_count + 1}/{max_retries}...")
            conn = psycopg2.connect(settings.database_url)

            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            db_version = cursor.fetchone()[0]
            logger.info(f"[성공] PostgreSQL 버전: {db_version}")

            try:
                cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
                vector_ext = cursor.fetchone()
                if vector_ext:
                    logger.info("[성공] pgvector 확장이 설치되어 있습니다.")
                else:
                    logger.warning("[경고] pgvector 확장이 설치되어 있지 않습니다.")
            except Exception as e:
                logger.warning(f"[경고] pgvector 확장 확인 실패: {e}")

            cursor.close()
            conn.close()

            logger.info("="*60)
            logger.info("[성공] ✅ Neon DB 연결 테스트 성공!")
            logger.info("="*60)

            return {
                "status": "success",
                "database_version": db_version,
                "connection_string": masked_final,
                "has_vector_extension": vector_ext is not None if 'vector_ext' in locals() else False
            }

        except psycopg2.OperationalError as exc:
            retry_count += 1
            error_msg = str(exc)
            logger.warning(f"[실패] 연결 실패 ({retry_count}/{max_retries}): {error_msg}")

            if retry_count < max_retries:
                logger.info(f"[대기] 2초 후 재시도...")
                time.sleep(2)
            else:
                logger.error("="*60)
                logger.error("[실패] ❌ Neon DB 연결 테스트 실패!")
                logger.error(f"[오류] 마지막 오류: {error_msg}")
                logger.error("="*60)

                return {
                    "status": "failed",
                    "error": error_msg,
                    "retries": retry_count
                }

        except Exception as exc:
            logger.error("="*60)
            logger.error(f"[오류] 예상치 못한 오류 발생: {exc}")
            logger.error("="*60)
            return {
                "status": "error",
                "error": str(exc)
            }

    return {
        "status": "failed",
        "error": "최대 재시도 횟수 초과",
        "retries": max_retries
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행되는 함수."""
    # 시작 시
    logger.info("="*60)
    logger.info("[시작] FastAPI RAG 애플리케이션 시작 중...")
    logger.info("="*60)

    # Neon DB 연결 테스트
    test_result = test_neon_db_connection()
    app.state.db_test_result = test_result

    logger.info("[완료] 애플리케이션 준비 완료!")
    yield

    # 종료 시
    logger.info("[종료] 애플리케이션 종료 중...")


# FastAPI 인스턴스 생성
app = FastAPI(
    title="RAG API Server",
    version="1.0.0",
    description="LangChain과 pgvector를 사용한 RAG API 서버",
    lifespan=lifespan,
)


# 미들웨어 설정 (CORS, 로깅, 에러 처리)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 전역 예외 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 핸들러 - 모든 예외를 캐치하여 로깅."""
    error_msg = str(exc)
    logger.error(f"[오류] 전역 예외 발생: {error_msg}")
    logger.error(f"[오류] 요청 경로: {request.url.path}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"서버 내부 오류: {error_msg}",
            "path": request.url.path,
        },
    )


# 라우터 등록 (API 엔드포인트 정의)
try:
    # 프로젝트 루트를 Python 경로에 추가
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Admin 라우터 등록
    from app.api.v1.admin import (
        consumer_router,
        order_router,
        product_router,
        email_router,
    )

    api_prefix = "/api/v1/admin"

    app.include_router(
        consumer_router.router,
        prefix=api_prefix+"/consumers",
        tags=["consumers"]
    )
    app.include_router(
        order_router.router,
        prefix=api_prefix+"/orders",
        tags=["orders"]
    )
    app.include_router(
        product_router.router,
        prefix=api_prefix+"/products",
        tags=["products"]
    )
    app.include_router(
        email_router.router,
        prefix=api_prefix+"/emails",
        tags=["emails"]
    )

    logger.info("[성공] 라우터 등록 완료")
except ImportError as e:
    logger.warning(f"[경고] 라우터 import 실패: {e}")
except Exception as e:
    logger.error(f"[오류] 라우터 등록 실패: {e}")


# 루트 엔드포인트
@app.get("/", tags=["root"])
async def root() -> dict:
    """루트 엔드포인트."""
    return {
        "message": "RAG API Server",
        "version": "1.0.0",
        "status": "running"
    }


# 헬스체크 엔드포인트
@app.get("/health", tags=["health"])
async def health() -> dict:
    """헬스체크 엔드포인트."""
    try:
        conn = psycopg2.connect(settings.database_url)
        conn.close()
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return {
        "status": "healthy",
        "version": "1.0.0",
        "database": db_status,
        "openai_configured": os.getenv("OPENAI_API_KEY") is not None,
    }


# ===== 메인 실행 =====
if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("[실행] FastAPI 서버 시작")
    logger.info("="*60)
    logger.info(f"[서버] http://127.0.0.1:8000 에서 실행됩니다")
    logger.info(f"[문서] http://127.0.0.1:8000/docs 에서 API 문서를 확인할 수 있습니다")
    logger.info("="*60 + "\n")

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.debug if hasattr(settings, "debug") else False,
    )
