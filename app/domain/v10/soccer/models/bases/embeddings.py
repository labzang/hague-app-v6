"""임베딩 벡터 SQLAlchemy 모델들.

RAG 및 시맨틱 검색을 위한 선수, 팀, 경기 일정, 경기장 임베딩 벡터를 저장하는 모델들.
KoElectra 모델(768 차원)을 사용하여 생성된 임베딩을 저장합니다.
"""

from typing import TYPE_CHECKING
from datetime import datetime
import numpy as np
from sqlalchemy import BigInteger, Text, ForeignKey, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from app.domain.shared.bases import Base

if TYPE_CHECKING:
    from app.domain.v10.soccer.models.bases.players import Player
    from app.domain.v10.soccer.models.bases.teams import Team
    from app.domain.v10.soccer.models.bases.schedules import Schedule
    from app.domain.v10.soccer.models.bases.stadiums import Stadium


class PlayerEmbedding(Base):
    """선수 임베딩 벡터를 저장하는 SQLAlchemy 모델.

    선수 정보(이름, 포지션, 국적 등)를 기반으로 생성된 임베딩 벡터를 저장합니다.
    pgvector의 HNSW 인덱스를 사용하여 빠른 유사도 검색을 지원합니다.

    Attributes:
        id: 임베딩 레코드 고유 식별자 (자동 증가)
        player_id: 선수 ID (외래 키 -> players.id)
        content: 임베딩 생성에 사용된 원본 텍스트 (예: 선수명 + 포지션 + 국적 등)
        embedding: 768차원 임베딩 벡터 (KoElectra)
        created_at: 레코드 생성 시간
    """

    __tablename__ = "players_embeddings"

    # 기본 키
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="임베딩 레코드 고유 식별자"
    )

    # 외래 키
    player_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False,
        comment="선수 ID"
    )

    # 임베딩 데이터
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="임베딩 생성에 사용된 원본 텍스트"
    )

    embedding: Mapped[np.ndarray] = mapped_column(
        Vector(768),
        nullable=False,
        comment="768차원 임베딩 벡터 (KoElectra)"
    )

    # 타임스탬프
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="레코드 생성 시간"
    )

    # 관계
    player: Mapped["Player"] = relationship(
        "Player",
        back_populates="embeddings"
    )

    def __repr__(self) -> str:
        """객체 문자열 표현."""
        return (
            f"<PlayerEmbedding(id={self.id}, "
            f"player_id={self.player_id}, "
            f"content_length={len(self.content) if self.content else 0})>"
        )


class TeamEmbedding(Base):
    """팀 임베딩 벡터를 저장하는 SQLAlchemy 모델.

    팀 정보를 기반으로 생성된 임베딩 벡터를 저장합니다.
    pgvector의 HNSW 인덱스를 사용하여 빠른 유사도 검색을 지원합니다.

    Attributes:
        id: 임베딩 레코드 고유 식별자 (자동 증가)
        team_id: 팀 ID (외래 키 -> teams.id)
        content: 임베딩 생성에 사용된 원본 텍스트
        embedding: 768차원 임베딩 벡터 (KoElectra)
        created_at: 레코드 생성 시간
    """

    __tablename__ = "teams_embeddings"

    # 기본 키
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="임베딩 레코드 고유 식별자"
    )

    # 외래 키
    team_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("teams.id", ondelete="CASCADE"),
        nullable=False,
        comment="팀 ID"
    )

    # 임베딩 데이터
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="임베딩 생성에 사용된 원본 텍스트"
    )

    embedding: Mapped[np.ndarray] = mapped_column(
        Vector(768),
        nullable=False,
        comment="768차원 임베딩 벡터 (KoElectra)"
    )

    # 타임스탬프
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="레코드 생성 시간"
    )

    # 관계
    team: Mapped["Team"] = relationship(
        "Team",
        back_populates="embeddings"
    )

    def __repr__(self) -> str:
        """객체 문자열 표현."""
        return (
            f"<TeamEmbedding(id={self.id}, "
            f"team_id={self.team_id}, "
            f"content_length={len(self.content) if self.content else 0})>"
        )


class ScheduleEmbedding(Base):
    """경기 일정 임베딩 벡터를 저장하는 SQLAlchemy 모델.

    경기 일정 정보를 기반으로 생성된 임베딩 벡터를 저장합니다.
    pgvector의 HNSW 인덱스를 사용하여 빠른 유사도 검색을 지원합니다.

    Attributes:
        id: 임베딩 레코드 고유 식별자 (자동 증가)
        schedule_id: 경기 일정 ID (외래 키 -> schedules.id)
        content: 임베딩 생성에 사용된 원본 텍스트
        embedding: 768차원 임베딩 벡터 (KoElectra)
        created_at: 레코드 생성 시간
    """

    __tablename__ = "schedules_embeddings"

    # 기본 키
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="임베딩 레코드 고유 식별자"
    )

    # 외래 키
    schedule_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("schedules.id", ondelete="CASCADE"),
        nullable=False,
        comment="경기 일정 ID"
    )

    # 임베딩 데이터
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="임베딩 생성에 사용된 원본 텍스트"
    )

    embedding: Mapped[np.ndarray] = mapped_column(
        Vector(768),
        nullable=False,
        comment="768차원 임베딩 벡터 (KoElectra)"
    )

    # 타임스탬프
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="레코드 생성 시간"
    )

    # 관계
    schedule: Mapped["Schedule"] = relationship(
        "Schedule",
        back_populates="embeddings"
    )

    def __repr__(self) -> str:
        """객체 문자열 표현."""
        return (
            f"<ScheduleEmbedding(id={self.id}, "
            f"schedule_id={self.schedule_id}, "
            f"content_length={len(self.content) if self.content else 0})>"
        )


class StadiumEmbedding(Base):
    """경기장 임베딩 벡터를 저장하는 SQLAlchemy 모델.

    경기장 정보를 기반으로 생성된 임베딩 벡터를 저장합니다.
    pgvector의 HNSW 인덱스를 사용하여 빠른 유사도 검색을 지원합니다.

    Attributes:
        id: 임베딩 레코드 고유 식별자 (자동 증가)
        stadium_id: 경기장 ID (외래 키 -> stadiums.id)
        content: 임베딩 생성에 사용된 원본 텍스트
        embedding: 768차원 임베딩 벡터 (KoElectra)
        created_at: 레코드 생성 시간
    """

    __tablename__ = "stadiums_embeddings"

    # 기본 키
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="임베딩 레코드 고유 식별자"
    )

    # 외래 키
    stadium_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("stadiums.id", ondelete="CASCADE"),
        nullable=False,
        comment="경기장 ID"
    )

    # 임베딩 데이터
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="임베딩 생성에 사용된 원본 텍스트"
    )

    embedding: Mapped[np.ndarray] = mapped_column(
        Vector(768),
        nullable=False,
        comment="768차원 임베딩 벡터 (KoElectra)"
    )

    # 타임스탬프
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="레코드 생성 시간"
    )

    # 관계
    stadium: Mapped["Stadium"] = relationship(
        "Stadium",
        back_populates="embeddings"
    )

    def __repr__(self) -> str:
        """객체 문자열 표현."""
        return (
            f"<StadiumEmbedding(id={self.id}, "
            f"stadium_id={self.stadium_id}, "
            f"content_length={len(self.content) if self.content else 0})>"
        )

