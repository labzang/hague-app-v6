"""경기 일정(Schedule) SQLAlchemy 모델."""

from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship

from app.domain.shared.bases import Base


class Schedule(Base):
    """경기 일정 정보를 저장하는 SQLAlchemy 모델.

    Attributes:
        sche_date: 경기 일자 (PK, 복합키의 일부)
        stadium_id: 경기장 ID (PK, 복합키의 일부, FK -> stadium.stadium_id)
        gubun: 구분
        hometeam_id: 홈팀 ID
        awayteam_id: 원정팀 ID
        home_score: 홈팀 점수
        away_score: 원정팀 점수
    """

    __tablename__ = "schedule"

    # 복합 기본 키
    sche_date = Column(
        String(10),
        primary_key=True,
        comment="경기 일자"
    )

    stadium_id = Column(
        String(10),
        ForeignKey("stadium.stadium_id"),
        primary_key=True,
        comment="경기장 ID"
    )

    # 경기 정보
    gubun = Column(
        String(10),
        nullable=True,
        comment="구분"
    )

    hometeam_id = Column(
        String(10),
        nullable=True,
        comment="홈팀 ID"
    )

    awayteam_id = Column(
        String(10),
        nullable=True,
        comment="원정팀 ID"
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

