"""팀(Team) SQLAlchemy 모델."""

from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.orm import relationship

from app.domain.shared.bases import Base


class Team(Base):
    """팀 정보를 저장하는 SQLAlchemy 모델.

    Attributes:
        team_id: 팀 고유 식별자 (PK)
        region_name: 지역명
        team_name: 팀명
        e_team_name: 영문 팀명
        orig_yyyy: 창단년도
        zip_code1: 우편번호1
        zip_code2: 우편번호2
        address: 주소
        ddd: 지역번호
        tel: 전화번호
        fax: 팩스번호
        homepage: 홈페이지
        owner: 구단주
        stadium_id: 경기장 ID (FK -> stadium.stadium_id)
    """

    __tablename__ = "team"

    # 기본 키
    team_id = Column(
        String(10),
        primary_key=True,
        comment="팀 고유 식별자"
    )

    # 팀 정보
    region_name = Column(
        String(10),
        nullable=True,
        comment="지역명"
    )

    team_name = Column(
        String(40),
        nullable=True,
        comment="팀명"
    )

    e_team_name = Column(
        String(50),
        nullable=True,
        comment="영문 팀명"
    )

    orig_yyyy = Column(
        String(10),
        nullable=True,
        comment="창단년도"
    )

    zip_code1 = Column(
        String(10),
        nullable=True,
        comment="우편번호1"
    )

    zip_code2 = Column(
        String(10),
        nullable=True,
        comment="우편번호2"
    )

    address = Column(
        String(80),
        nullable=True,
        comment="주소"
    )

    ddd = Column(
        String(10),
        nullable=True,
        comment="지역번호"
    )

    tel = Column(
        String(10),
        nullable=True,
        comment="전화번호"
    )

    fax = Column(
        String(10),
        nullable=True,
        comment="팩스번호"
    )

    homepage = Column(
        String(50),
        nullable=True,
        comment="홈페이지"
    )

    owner = Column(
        String(10),
        nullable=True,
        comment="구단주"
    )

    # 외래 키
    stadium_id = Column(
        String(10),
        ForeignKey("stadium.stadium_id"),
        nullable=True,
        comment="경기장 ID"
    )

    # 관계
    stadium = relationship(
        "Stadium",
        back_populates="teams"
    )

    players = relationship(
        "Player",
        back_populates="team",
        cascade="all, delete-orphan"
    )

