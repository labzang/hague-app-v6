"""
벡터 관련 SQLAlchemy 모델들
판독 에이전트 결과 및 임베딩 벡터를 저장하는 데이터베이스 모델들
"""

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from typing import Optional, Dict, Any
import uuid

# SQLAlchemy Base 클래스 생성
Base = declarative_base()


class VerdictVectorRecord(Base):
    """판독 에이전트 결과 및 임베딩 벡터를 저장하는 모델

    이메일 분석 결과와 임베딩 벡터를 PostgreSQL에 저장하여
    유사도 검색 및 히스토리 추적을 가능하게 합니다.
    """

    __tablename__ = "verdict_vector_records"

    # 기본 키
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="레코드 고유 식별자"
    )

    # 세션 정보
    session_id = Column(
        String(255),
        nullable=False,
        index=True,
        comment="처리 세션 ID"
    )

    # 이메일 정보
    email_subject = Column(
        Text,
        nullable=False,
        comment="이메일 제목"
    )

    email_content = Column(
        Text,
        nullable=False,
        comment="이메일 내용"
    )

    email_sender = Column(
        String(255),
        nullable=True,
        comment="이메일 발신자"
    )

    # 벡터 임베딩 (pgvector 사용 시)
    # PostgreSQL의 pgvector 확장을 사용하려면 다음과 같이 정의:
    # from pgvector.sqlalchemy import Vector
    # embedding_vector = Column(Vector(384), nullable=True)

    # 예시: JSON 형태로 임베딩 벡터 저장 (pgvector 미사용 시)
    embedding_vector = Column(
        JSON,
        nullable=True,
        comment="이메일 임베딩 벡터 (JSON 배열 형태)"
    )

    embedding_dimension = Column(
        Integer,
        nullable=True,
        comment="임베딩 벡터 차원"
    )

    # 판독 결과
    verdict = Column(
        String(50),
        nullable=False,
        index=True,
        comment="판정 결과: spam, normal, uncertain"
    )

    confidence_score = Column(
        Float,
        nullable=False,
        comment="신뢰도 점수 (0.0 ~ 1.0)"
    )

    koelectra_result = Column(
        JSON,
        nullable=True,
        comment="KoELECTRA 분석 결과 (JSON 형태)"
    )

    exaone_analysis = Column(
        Text,
        nullable=True,
        comment="EXAONE 분석 결과"
    )

    analysis_type = Column(
        String(50),
        nullable=True,
        comment="분석 타입: detailed, quick"
    )

    # 메타데이터
    metadata_json = Column(
        JSON,
        nullable=True,
        comment="추가 메타데이터 (JSON 형태)"
    )

    processing_steps = Column(
        JSON,
        nullable=True,
        comment="처리 단계 목록 (JSON 배열 형태)"
    )

    # 타임스탬프
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="레코드 생성 시간"
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="레코드 수정 시간"
    )

    # 인덱스 정의
    __table_args__ = (
        # 세션 ID와 판정 결과 복합 인덱스
        Index('idx_session_verdict', 'session_id', 'verdict'),
        # 신뢰도 점수 인덱스 (범위 쿼리용)
        Index('idx_confidence_score', 'confidence_score'),
        # 생성 시간 인덱스 (시간 기반 쿼리용)
        Index('idx_created_at', 'created_at'),
    )

    def __repr__(self) -> str:
        """객체 문자열 표현"""
        return (
            f"<VerdictVectorRecord(id={self.id}, "
            f"session_id={self.session_id}, "
            f"verdict={self.verdict}, "
            f"confidence={self.confidence_score:.3f})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """모델을 딕셔너리로 변환"""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "email_subject": self.email_subject,
            "email_content": self.email_content,
            "email_sender": self.email_sender,
            "embedding_vector": self.embedding_vector,
            "embedding_dimension": self.embedding_dimension,
            "verdict": self.verdict,
            "confidence_score": self.confidence_score,
            "koelectra_result": self.koelectra_result,
            "exaone_analysis": self.exaone_analysis,
            "analysis_type": self.analysis_type,
            "metadata_json": self.metadata_json,
            "processing_steps": self.processing_steps,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
