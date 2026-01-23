"""상품(Products) SQLAlchemy 모델."""

from sqlalchemy import Column, Integer, String, Text, Boolean, CheckConstraint, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.domain.shared.bases import Base


class Product(Base):
    """상품 정보를 저장하는 SQLAlchemy 모델.

    Attributes:
        id: 상품 고유 식별자 (자동 증가)
        name: 상품명
        description: 상품 설명 (임베딩 원문용)
        price: 가격 (원 단위, 0 이상)
        category: 카테고리
        brand: 브랜드
        is_active: 판매 여부
        created_at: 생성 일시
        updated_at: 수정 일시
    """

    __tablename__ = "products"

    # 기본 키
    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="상품 고유 식별자"
    )

    # 상품 정보
    name = Column(
        Text,
        nullable=False,
        comment="상품명"
    )

    description = Column(
        Text,
        nullable=True,
        comment="상품 설명 (임베딩 원문용)"
    )

    price = Column(
        Integer,
        nullable=False,
        comment="가격 (원 단위)"
    )

    category = Column(
        String(100),
        nullable=True,
        comment="카테고리"
    )

    brand = Column(
        String(100),
        nullable=True,
        comment="브랜드"
    )

    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="판매 여부"
    )

    # 타임스탬프
    created_at = Column(
        TIMESTAMP,
        server_default=func.now(),
        nullable=False,
        comment="생성 일시"
    )

    updated_at = Column(
        TIMESTAMP,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="수정 일시"
    )

    # 제약조건
    __table_args__ = (
        CheckConstraint("price >= 0", name="check_price_non_negative"),
    )

    # 관계
    orders = relationship(
        "Order",
        back_populates="product",
        cascade="all, delete-orphan",
        comment="주문 목록"
    )

    def __repr__(self) -> str:
        """객체 표현 문자열."""
        return f"<Product(id={self.id}, name='{self.name}', price={self.price})>"

