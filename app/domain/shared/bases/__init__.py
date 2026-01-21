"""SQLAlchemy Base 모델 정의."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """공통 SQLAlchemy Base 클래스.

    SQLAlchemy 2.0 스타일의 DeclarativeBase를 상속받은 Base 클래스입니다.
    모든 모델은 이 Base 클래스를 상속받아야 합니다.
    """
    pass

