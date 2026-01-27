"""주문(Order) SQLAlchemy 모델 (교차 엔티티)."""

from sqlalchemy import (
    Column,
    Integer,
    TIMESTAMP,
    ForeignKey,
    Enum as SQLEnum,
    CheckConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum

from app.domain.shared.bases import Base


class OrderStatus(str, Enum):
    """주문 상태 열거형."""

    PENDING = "pending"  # 대기 중
    CONFIRMED = "confirmed"  # 확인됨
    PROCESSING = "processing"  # 처리 중
    SHIPPED = "shipped"  # 배송 중
    DELIVERED = "delivered"  # 배송 완료
    CANCELLED = "cancelled"  # 취소됨


class Order(Base):
    """주문 정보를 저장하는 SQLAlchemy 모델 (교차 엔티티).

    Product와 Consumer를 연결하는 교차 엔티티입니다.

    Attributes:
        id: 주문 고유 식별자 (자동 증가)
        consumer_id: 소비자 ID (외래키)
        product_id: 상품 ID (외래키)
        quantity: 주문 수량
        unit_price: 단가 (주문 시점의 가격)
        total_price: 총 가격 (quantity * unit_price)
        status: 주문 상태
        order_date: 주문 일시
        created_at: 생성 일시
        updated_at: 수정 일시
        consumer: 소비자 (관계)
        product: 상품 (관계)
    """

    __tablename__ = "orders"

    # 기본 키
    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="주문 고유 식별자"
    )

    # 외래키
    consumer_id = Column(
        Integer,
        ForeignKey("consumers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="소비자 ID"
    )

    product_id = Column(
        Integer,
        ForeignKey("products.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        comment="상품 ID"
    )

    # 주문 정보
    quantity = Column(
        Integer,
        nullable=False,
        default=1,
        comment="주문 수량"
    )

    unit_price = Column(
        Integer,
        nullable=False,
        comment="단가 (주문 시점의 가격)"
    )

    total_price = Column(
        Integer,
        nullable=False,
        comment="총 가격 (quantity * unit_price)"
    )

    status = Column(
        SQLEnum(OrderStatus, name="order_status"),
        nullable=False,
        default=OrderStatus.PENDING,
        comment="주문 상태"
    )

    order_date = Column(
        TIMESTAMP,
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="주문 일시"
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
        CheckConstraint("quantity > 0", name="check_quantity_positive"),
        CheckConstraint("unit_price >= 0", name="check_unit_price_non_negative"),
        CheckConstraint("total_price >= 0", name="check_total_price_non_negative"),
    )

    # 관계
    consumer = relationship(
        "Consumer",
        back_populates="orders",
        comment="소비자"
    )

    product = relationship(
        "Product",
        back_populates="orders",
        comment="상품"
    )

    def __repr__(self) -> str:
        """객체 표현 문자열."""
        return (
            f"<Order(id={self.id}, consumer_id={self.consumer_id}, "
            f"product_id={self.product_id}, quantity={self.quantity}, "
            f"status='{self.status.value}')>"
        )

