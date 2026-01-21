"""상품(Product) Pydantic 모델."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ProductModel(BaseModel):
    """상품 정보를 전송하기 위한 Pydantic 모델.

    SQLAlchemy Product 모델의 transfer 객체입니다.
    """

    id: Optional[int] = Field(None, description="상품 고유 식별자")
    name: str = Field(..., description="상품명", min_length=1)
    description: Optional[str] = Field(None, description="상품 설명 (임베딩 원문용)")
    price: int = Field(..., description="가격 (원 단위)", ge=0)
    category: Optional[str] = Field(None, description="카테고리", max_length=100)
    brand: Optional[str] = Field(None, description="브랜드", max_length=100)
    is_active: bool = Field(True, description="판매 여부")
    created_at: Optional[datetime] = Field(None, description="생성 일시")
    updated_at: Optional[datetime] = Field(None, description="수정 일시")

    class Config:
        """Pydantic 설정."""

        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ProductCreateModel(BaseModel):
    """상품 생성 요청 모델."""

    name: str = Field(..., description="상품명", min_length=1)
    description: Optional[str] = Field(None, description="상품 설명 (임베딩 원문용)")
    price: int = Field(..., description="가격 (원 단위)", ge=0)
    category: Optional[str] = Field(None, description="카테고리", max_length=100)
    brand: Optional[str] = Field(None, description="브랜드", max_length=100)
    is_active: bool = Field(True, description="판매 여부")


class ProductUpdateModel(BaseModel):
    """상품 수정 요청 모델."""

    name: Optional[str] = Field(None, description="상품명", min_length=1)
    description: Optional[str] = Field(None, description="상품 설명 (임베딩 원문용)")
    price: Optional[int] = Field(None, description="가격 (원 단위)", ge=0)
    category: Optional[str] = Field(None, description="카테고리", max_length=100)
    brand: Optional[str] = Field(None, description="브랜드", max_length=100)
    is_active: Optional[bool] = Field(None, description="판매 여부")

