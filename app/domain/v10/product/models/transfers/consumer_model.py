"""소비자(Consumer) Pydantic 모델."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class ConsumerModel(BaseModel):
    """소비자 정보를 전송하기 위한 Pydantic 모델.

    SQLAlchemy Consumer 모델의 transfer 객체입니다.
    """

    id: Optional[int] = Field(None, description="소비자 고유 식별자")
    name: str = Field(..., description="소비자 이름", min_length=1, max_length=100)
    email: EmailStr = Field(..., description="이메일 주소")
    phone: Optional[str] = Field(None, description="전화번호", max_length=20)
    address: Optional[str] = Field(None, description="배송 주소")
    created_at: Optional[datetime] = Field(None, description="생성 일시")
    updated_at: Optional[datetime] = Field(None, description="수정 일시")

    class Config:
        """Pydantic 설정."""

        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ConsumerCreateModel(BaseModel):
    """소비자 생성 요청 모델."""

    name: str = Field(..., description="소비자 이름", min_length=1, max_length=100)
    email: EmailStr = Field(..., description="이메일 주소")
    phone: Optional[str] = Field(None, description="전화번호", max_length=20)
    address: Optional[str] = Field(None, description="배송 주소")


class ConsumerUpdateModel(BaseModel):
    """소비자 수정 요청 모델."""

    name: Optional[str] = Field(None, description="소비자 이름", min_length=1, max_length=100)
    email: Optional[EmailStr] = Field(None, description="이메일 주소")
    phone: Optional[str] = Field(None, description="전화번호", max_length=20)
    address: Optional[str] = Field(None, description="배송 주소")

