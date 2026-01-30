from sqlalchemy import Column, BigInteger, Text, ForeignKey, TIMESTAMP, func, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Vector
from pgvector import Vector as PgVector
from app.domain.shared.bases import Base

Base = declarative_base()

class ScheduleEmbedding(Base):
    """임베딩 레코드 모델.

    Attributes:
        id: 임베딩 레코드 고유 식별자 (PK, BigInt)
        schedule_id: 경기 일정 ID (FK -> schedules.id)
        content: 원본 텍스트 데이터 (Text)
        embedding: 768차원 KoElectra 임베딩 벡터 (Vector)
        created_at: 레코드 생성 시간 (TIMESTAMP)
    """
    __tablename__ = "schedules_embeddings"

    # 기본 키
    id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="임베딩 레코드 고유 식별자"
    )

    # 외래 키
    schedule_id = Column(
        BigInteger,
        ForeignKey("schedules.id"),
        nullable=False,
        ondelete="CASCADE",
        comment="경기 일정 ID"
    )

    content = Column(
        Text,
        nullable=False,
        comment="임베딩 생성에 사용된 원본 텍스트"
    )

    embedding = Column(
        PgVector(dim=768),
        nullable=False,
        comment='768차원 KoElectra 임베딩 벡터'
    )

    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment='레코드 생성 시간'
    )

    # 관계
    schedule = relationship(
        "Schedule",
        back_populates="embeddings"
    )

class Schedule(Base):
    """경기 일정 정보를 저장하는 SQLAlchemy 모델.

    Attributes:
        id: 경기 일정 고유 식별자 (PK, BigInt)
        stadium_id: 경기장 ID (FK -> stadiums.id)
        hometeam_id: 홈팀 ID (FK -> teams.id)
        awayteam_id: 원정팀 ID (FK -> teams.id)
        stadium_code: 경기장 코드
        sche_date: 경기 일자
        gubun: 구분
        hometeam_code: 홈팀 코드
        awayteam_code: 원정팀 코드
        home_score: 홈팀 점수
        away_score: 원정팀 점수
    """

    __tablename__ = "schedules"

    # 기본 키
    id = Column(
        BigInteger,
        primary_key=True,
        comment="경기 일정 고유 식별자"
    )

    # 외래 키
    stadium_id = Column(
        BigInteger,
        ForeignKey("stadiums.id"),
        nullable=True,
        comment="경기장 ID"
    )

    hometeam_id = Column(
        BigInteger,
        ForeignKey("teams.id"),
        nullable=True,
        comment="홈팀 ID"
    )

    awayteam_id = Column(
        BigInteger,
        ForeignKey("teams.id"),
        nullable=True,
        comment="원정팀 ID"
    )

    # 경기 정보
    stadium_code = Column(
        String(10),
        nullable=True,
        comment="경기장 코드"
    )

    sche_date = Column(
        String(10),
        nullable=True,
        comment="경기 일자"
    )

    gubun = Column(
        String(10),
        nullable=True,
        comment="구분"
    )

    hometeam_code = Column(
        String(10),
        nullable=True,
        comment="홈팀 코드"
    )

    awayteam_code = Column(
        String(10),
        nullable=True,
        comment="원정팀 코드"
    )

    home_score = Column(
        Integer,
        nullable=True,
        comment="홈팀 점수"
    )

    away_score = Column(
        Integer,
        nullable=True,
        comment="원정팀 점수"
    )

    # 관계
    stadium = relationship(
        "Stadium",
        back_populates="schedules"
    )

    hometeam = relationship(
        "Team",
        foreign_keys=[hometeam_id],
        backref="home_schedules"
    )

    awayteam = relationship(
        "Team",
        foreign_keys=[awayteam_id],
        backref="away_schedules"
    )

    embeddings = relationship(
        "ScheduleEmbedding",
        back_populates="schedule",