"""상품(Product) 상태 머신."""

from enum import Enum
from typing import Optional, Set
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class ProductStatus(str, Enum):
    """상품 상태 열거형."""

    ACTIVE = "active"  # 판매 중
    INACTIVE = "inactive"  # 판매 중지
    OUT_OF_STOCK = "out_of_stock"  # 품절
    DISCONTINUED = "discontinued"  # 단종
    DELETED = "deleted"  # 삭제됨


class ProductState(BaseModel):
    """상품 상태 머신 클래스.

    상품의 상태를 관리하고 상태 전이를 검증합니다.
    """

    status: ProductStatus = Field(
        default=ProductStatus.ACTIVE,
        description="상품 상태"
    )
    is_active: bool = Field(
        default=True,
        description="판매 여부 (is_active 필드와 동기화)"
    )
    last_status_change: Optional[datetime] = Field(
        None,
        description="마지막 상태 변경 일시"
    )
    status_history: list[dict] = Field(
        default_factory=list,
        description="상태 변경 이력"
    )

    # 상태 전이 규칙
    _valid_transitions: dict[ProductStatus, Set[ProductStatus]] = {
        ProductStatus.ACTIVE: {
            ProductStatus.INACTIVE,
            ProductStatus.OUT_OF_STOCK,
            ProductStatus.DISCONTINUED,
            ProductStatus.DELETED
        },
        ProductStatus.INACTIVE: {
            ProductStatus.ACTIVE,
            ProductStatus.DELETED
        },
        ProductStatus.OUT_OF_STOCK: {
            ProductStatus.ACTIVE,
            ProductStatus.INACTIVE,
            ProductStatus.DISCONTINUED,
            ProductStatus.DELETED
        },
        ProductStatus.DISCONTINUED: {
            ProductStatus.DELETED
        },
        ProductStatus.DELETED: set()  # 삭제 상태에서는 전이 불가
    }

    @field_validator("is_active", mode="before")
    @classmethod
    def sync_is_active(cls, v: bool) -> bool:
        """is_active 필드를 status와 동기화."""
        # transition_to 메서드에서 직접 동기화하므로 여기서는 기본값 반환
        return v

    def can_transition_to(self, new_status: ProductStatus) -> bool:
        """상태 전이가 가능한지 확인.

        Args:
            new_status: 전이하려는 새로운 상태

        Returns:
            전이 가능 여부
        """
        if new_status == self.status:
            return True  # 같은 상태로의 전이는 허용

        allowed_transitions = self._valid_transitions.get(self.status, set())
        return new_status in allowed_transitions

    def transition_to(self, new_status: ProductStatus, reason: Optional[str] = None) -> bool:
        """상태를 전이.

        Args:
            new_status: 전이하려는 새로운 상태
            reason: 상태 변경 사유

        Returns:
            전이 성공 여부

        Raises:
            ValueError: 유효하지 않은 상태 전이인 경우
        """
        if not self.can_transition_to(new_status):
            raise ValueError(
                f"상태 전이 불가: {self.status.value} -> {new_status.value}"
            )

        # 상태 이력 기록
        self.status_history.append({
            "from_status": self.status.value,
            "to_status": new_status.value,
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        })

        self.status = new_status
        self.last_status_change = datetime.now()

        # is_active 필드 동기화
        self.is_active = new_status == ProductStatus.ACTIVE

        return True

    def is_sellable(self) -> bool:
        """판매 가능한 상태인지 확인."""
        return self.status == ProductStatus.ACTIVE and self.is_active

    def is_deleted(self) -> bool:
        """삭제 상태인지 확인."""
        return self.status == ProductStatus.DELETED

    def can_be_ordered(self) -> bool:
        """주문 가능한 상태인지 확인."""
        return self.status in {
            ProductStatus.ACTIVE,
            ProductStatus.OUT_OF_STOCK  # 품절이어도 주문은 가능 (재입고 대기)
        } and not self.is_deleted()

    class Config:
        """Pydantic 설정."""

        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

