"""주문(Order) API 라우터."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.domain.admin.models.order_model import (
    OrderModel,
    OrderCreateModel,
    OrderUpdateModel,
    OrderDetailModel,
)
from app.domain.admin.orchestrators.order_flow import OrderFlow

router = APIRouter()


class OrderRequest(BaseModel):
    """주문 요청 모델."""
    action: str  # "create", "update", "get", "list", "delete", "cancel"
    data: Optional[dict] = None
    order_id: Optional[int] = None
    use_policy: bool = False  # True: 정책 기반, False: 규칙 기반


@router.post("/", response_model=dict)
async def handle_order_request(request: OrderRequest):
    """주문 요청 처리 엔드포인트.

    규칙 기반 또는 정책 기반으로 요청을 처리합니다.
    """
    try:
        flow = OrderFlow()
        result = await flow.process_request(
            action=request.action,
            data=request.data or {},
            order_id=request.order_id,
            use_policy=request.use_policy
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create", response_model=OrderModel)
async def create_order(order: OrderCreateModel, use_policy: bool = False):
    """주문 생성."""
    try:
        flow = OrderFlow()
        result = await flow.process_request(
            action="create",
            data=order.model_dump(),
            use_policy=use_policy
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{order_id}", response_model=OrderDetailModel)
async def get_order(order_id: int, use_policy: bool = False):
    """주문 조회."""
    try:
        flow = OrderFlow()
        result = await flow.process_request(
            action="get",
            order_id=order_id,
            use_policy=use_policy
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{order_id}", response_model=OrderModel)
async def update_order(
    order_id: int,
    order: OrderUpdateModel,
    use_policy: bool = False
):
    """주문 수정."""
    try:
        flow = OrderFlow()
        result = await flow.process_request(
            action="update",
            data=order.model_dump(exclude_unset=True),
            order_id=order_id,
            use_policy=use_policy
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[OrderModel])
async def list_orders(
    use_policy: bool = False,
    consumer_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """주문 목록 조회."""
    try:
        flow = OrderFlow()
        data = {"limit": limit, "offset": offset}
        if consumer_id:
            data["consumer_id"] = consumer_id
        if status:
            data["status"] = status

        result = await flow.process_request(
            action="list",
            data=data,
            use_policy=use_policy
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{order_id}", response_model=dict)
async def delete_order(order_id: int, use_policy: bool = False):
    """주문 삭제."""
    try:
        flow = OrderFlow()
        result = await flow.process_request(
            action="delete",
            order_id=order_id,
            use_policy=use_policy
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{order_id}/cancel", response_model=OrderModel)
async def cancel_order(order_id: int, use_policy: bool = False):
    """주문 취소."""
    try:
        flow = OrderFlow()
        result = await flow.process_request(
            action="cancel",
            order_id=order_id,
            use_policy=use_policy
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

