"""소비자(Consumer) 상태 머신."""

from enum import Enum
from typing import Optional, Set
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class ConsumerStatus(str, Enum):
    """소비자 상태 열거형."""

    ACTIVE = "active"  # 활성
    INACTIVE = "inactive"  # 비활성
    SUSPENDED = "suspended"  # 정지됨
    DELETED = "deleted"  # 삭제됨


class ConsumerState(BaseModel):
    """소비자 상태 머신 클래스.

    소비자의 상태를 관리하고 상태 전이를 검증합니다.
    """

    status: ConsumerStatus = Field(
        default=ConsumerStatus.ACTIVE,
        description="소비자 상태"
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
    _valid_transitions: dict[ConsumerStatus, Set[ConsumerStatus]] = {
        ConsumerStatus.ACTIVE: {
            ConsumerStatus.INACTIVE,
            ConsumerStatus.SUSPENDED,
            ConsumerStatus.DELETED
        },
        ConsumerStatus.INACTIVE: {
            ConsumerStatus.ACTIVE,
            ConsumerStatus.DELETED
        },
        ConsumerStatus.SUSPENDED: {
            ConsumerStatus.ACTIVE,
            ConsumerStatus.DELETED
        },
        ConsumerStatus.DELETED: set()  # 삭제 상태에서는 전이 불가
    }

    def can_transition_to(self, new_status: ConsumerStatus) -> bool:
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

    def transition_to(self, new_status: ConsumerStatus, reason: Optional[str] = None) -> bool:
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
        return True

    def is_active(self) -> bool:
        """활성 상태인지 확인."""
        return self.status == ConsumerStatus.ACTIVE

    def is_deleted(self) -> bool:
        """삭제 상태인지 확인."""
        return self.status == ConsumerStatus.DELETED

    class Config:
        """Pydantic 설정."""

        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

