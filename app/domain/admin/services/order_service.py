"""주문(Order) 규칙 기반 서비스."""

import logging
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.domain.admin.bases.orders import Order, OrderStatus
from app.domain.admin.bases.consumers import Consumer
from app.domain.admin.bases.products import Product
from app.domain.admin.models.order_model import (
    OrderModel,
    OrderCreateModel,
    OrderUpdateModel,
    OrderDetailModel,
)

logger = logging.getLogger(__name__)


class OrderService:
    """주문 규칙 기반 서비스.

    규칙 기반 로직으로 주문 CRUD 작업을 수행합니다.
    """

    def __init__(self):
        """OrderService 초기화."""
        # 데이터베이스 세션 생성
        engine = create_engine(settings.database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        self.SessionLocal = SessionLocal
        logger.info("[서비스] OrderService 초기화 완료")

    def _get_session(self) -> Session:
        """데이터베이스 세션 반환."""
        return self.SessionLocal()

    async def create_order(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """주문 생성 (규칙 기반).

        Args:
            data: 주문 생성 데이터

        Returns:
            생성된 주문 정보
        """
        logger.info(f"[서비스] 주문 생성 - data: {data}")

        # 규칙 기반 검증
        if not data.get("consumer_id"):
            raise ValueError("consumer_id는 필수입니다")
        if not data.get("product_id"):
            raise ValueError("product_id는 필수입니다")
        if not data.get("quantity") or data["quantity"] <= 0:
            raise ValueError("quantity는 0보다 커야 합니다")
        if not data.get("unit_price") or data["unit_price"] < 0:
            raise ValueError("unit_price는 0 이상이어야 합니다")

        # total_price 자동 계산
        if "total_price" not in data:
            data["total_price"] = data["quantity"] * data["unit_price"]

        session = self._get_session()
        try:
            # 소비자와 상품 존재 확인
            consumer = session.query(Consumer).filter(Consumer.id == data["consumer_id"]).first()
            if not consumer:
                raise ValueError(f"소비자를 찾을 수 없습니다: {data['consumer_id']}")

            product = session.query(Product).filter(Product.id == data["product_id"]).first()
            if not product:
                raise ValueError(f"상품을 찾을 수 없습니다: {data['product_id']}")

            # 주문 생성
            order = Order(
                consumer_id=data["consumer_id"],
                product_id=data["product_id"],
                quantity=data["quantity"],
                unit_price=data["unit_price"],
                total_price=data["total_price"],
                status=OrderStatus(data.get("status", OrderStatus.PENDING.value)),
            )
            session.add(order)
            session.commit()
            session.refresh(order)

            result = OrderModel.model_validate(order).model_dump()
            logger.info(f"[서비스] 주문 생성 완료 - id: {order.id}")
            return result
        except Exception as e:
            session.rollback()
            logger.error(f"[서비스] 주문 생성 실패: {e}")
            raise
        finally:
            session.close()

    async def update_order(
        self,
        order_id: int,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """주문 수정 (규칙 기반).

        Args:
            order_id: 주문 ID
            data: 수정할 데이터

        Returns:
            수정된 주문 정보
        """
        logger.info(f"[서비스] 주문 수정 - id: {order_id}, data: {data}")

        session = self._get_session()
        try:
            order = session.query(Order).filter(Order.id == order_id).first()
            if not order:
                raise ValueError(f"주문을 찾을 수 없습니다: {order_id}")

            # 규칙 기반 업데이트
            if "quantity" in data:
                if data["quantity"] <= 0:
                    raise ValueError("quantity는 0보다 커야 합니다")
                order.quantity = data["quantity"]
                # total_price 자동 재계산
                if "unit_price" in data:
                    order.total_price = data["quantity"] * data["unit_price"]
                else:
                    order.total_price = data["quantity"] * order.unit_price

            if "unit_price" in data:
                if data["unit_price"] < 0:
                    raise ValueError("unit_price는 0 이상이어야 합니다")
                order.unit_price = data["unit_price"]
                # total_price 자동 재계산
                order.total_price = order.quantity * data["unit_price"]

            if "total_price" in data:
                if data["total_price"] < 0:
                    raise ValueError("total_price는 0 이상이어야 합니다")
                order.total_price = data["total_price"]

            if "status" in data:
                order.status = OrderStatus(data["status"])

            session.commit()
            session.refresh(order)

            result = OrderModel.model_validate(order).model_dump()
            logger.info(f"[서비스] 주문 수정 완료 - id: {order_id}")
            return result
        except Exception as e:
            session.rollback()
            logger.error(f"[서비스] 주문 수정 실패: {e}")
            raise
        finally:
            session.close()

    async def get_order(self, order_id: int) -> Dict[str, Any]:
        """주문 조회 (규칙 기반).

        Args:
            order_id: 주문 ID

        Returns:
            주문 상세 정보
        """
        logger.info(f"[서비스] 주문 조회 - id: {order_id}")

        session = self._get_session()
        try:
            order = (
                session.query(Order)
                .join(Consumer, Order.consumer_id == Consumer.id)
                .join(Product, Order.product_id == Product.id)
                .filter(Order.id == order_id)
                .first()
            )
            if not order:
                raise ValueError(f"주문을 찾을 수 없습니다: {order_id}")

            # 상세 정보 구성
            result = OrderDetailModel(
                id=order.id,
                consumer_id=order.consumer_id,
                product_id=order.product_id,
                quantity=order.quantity,
                unit_price=order.unit_price,
                total_price=order.total_price,
                status=order.status,
                order_date=order.order_date,
                created_at=order.created_at,
                updated_at=order.updated_at,
                consumer_name=order.consumer.name,
                consumer_email=order.consumer.email,
                product_name=order.product.name,
                product_price=order.product.price,
            ).model_dump()

            return result
        except Exception as e:
            logger.error(f"[서비스] 주문 조회 실패: {e}")
            raise
        finally:
            session.close()

    async def list_orders(
        self,
        limit: int = 100,
        offset: int = 0,
        consumer_id: Optional[int] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """주문 목록 조회 (규칙 기반).

        Args:
            limit: 조회할 개수
            offset: 시작 위치
            consumer_id: 소비자 ID 필터
            status: 주문 상태 필터

        Returns:
            주문 목록
        """
        logger.info(f"[서비스] 주문 목록 조회 - limit: {limit}, offset: {offset}")

        session = self._get_session()
        try:
            query = session.query(Order)

            # 필터 적용
            if consumer_id:
                query = query.filter(Order.consumer_id == consumer_id)
            if status:
                query = query.filter(Order.status == OrderStatus(status))

            orders = query.order_by(Order.order_date.desc()).offset(offset).limit(limit).all()
            result = [OrderModel.model_validate(o).model_dump() for o in orders]
            return result
        except Exception as e:
            logger.error(f"[서비스] 주문 목록 조회 실패: {e}")
            raise
        finally:
            session.close()

    async def delete_order(self, order_id: int) -> Dict[str, Any]:
        """주문 삭제 (규칙 기반).

        Args:
            order_id: 주문 ID

        Returns:
            삭제 결과
        """
        logger.info(f"[서비스] 주문 삭제 - id: {order_id}")

        session = self._get_session()
        try:
            order = session.query(Order).filter(Order.id == order_id).first()
            if not order:
                raise ValueError(f"주문을 찾을 수 없습니다: {order_id}")

            # 규칙: 배송 완료된 주문은 삭제 불가
            if order.status == OrderStatus.DELIVERED:
                raise ValueError("배송 완료된 주문은 삭제할 수 없습니다")

            session.delete(order)
            session.commit()

            logger.info(f"[서비스] 주문 삭제 완료 - id: {order_id}")
            return {"status": "success", "message": f"주문 {order_id}가 삭제되었습니다"}
        except Exception as e:
            session.rollback()
            logger.error(f"[서비스] 주문 삭제 실패: {e}")
            raise
        finally:
            session.close()

    async def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """주문 취소 (규칙 기반).

        Args:
            order_id: 주문 ID

        Returns:
            취소된 주문 정보
        """
        logger.info(f"[서비스] 주문 취소 - id: {order_id}")

        session = self._get_session()
        try:
            order = session.query(Order).filter(Order.id == order_id).first()
            if not order:
                raise ValueError(f"주문을 찾을 수 없습니다: {order_id}")

            # 규칙: 배송 완료된 주문은 취소 불가
            if order.status == OrderStatus.DELIVERED:
                raise ValueError("배송 완료된 주문은 취소할 수 없습니다")

            # 규칙: 이미 취소된 주문은 취소 불가
            if order.status == OrderStatus.CANCELLED:
                raise ValueError("이미 취소된 주문입니다")

            order.status = OrderStatus.CANCELLED
            session.commit()
            session.refresh(order)

            result = OrderModel.model_validate(order).model_dump()
            logger.info(f"[서비스] 주문 취소 완료 - id: {order_id}")
            return result
        except Exception as e:
            session.rollback()
            logger.error(f"[서비스] 주문 취소 실패: {e}")
            raise
        finally:
            session.close()

