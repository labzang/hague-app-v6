# 핵심 코드 템플릿

## 1. BaseAgent 상속 패턴 (필수)

```python
# app/domain/admin/agents/{entity}_agent.py
from app.domain.admin.agents.base_agent import BaseAgent
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class EntityAgent(BaseAgent):
    def __init__(self, adapter_path: Optional[Path] = None):
        super().__init__(name="EntityAgent", instruction="...")
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        if adapter_path:
            self._load_model()

    def _load_model(self):
        """Fine-tuned 어댑터 로드 - 필수"""
        # PeftModel.from_pretrained() 사용
        # 베이스 모델 + LoRA 어댑터 로드

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """모델 미사용 시 폴백 처리 - 필수"""
        if self.model is None:
            return await self._fallback_execute(...)
        # 모델 사용 로직

    async def _fallback_execute(self, ...):
        """폴백 처리 - 필수"""
        return {"status": "success", "message": "기본 처리 완료 (모델 미사용)"}
```

## 2. Flow 오케스트레이터 패턴 (필수)

```python
# app/domain/admin/orchestrators/{entity}_flow.py
class EntityFlow:
    def __init__(self):
        self.service = None
        self.agent = None
        self.adapter_path = None
        self._load_adapter()  # 필수

    def _load_adapter(self):
        """어댑터 자동 로드 - 필수"""
        project_root = Path(__file__).parent.parent.parent.parent.parent
        adapter_base_path = project_root / "artifacts" / "fine-tuned-adapters" / "customer-service"
        # 가장 최근 run_* 디렉토리 찾기

    async def process_request(self, ..., use_policy: bool = False):
        """규칙/정책 분기 - 필수"""
        if use_policy:
            return await self._process_with_agent(...)
        else:
            return await self._process_with_service(...)
```

## 3. Service 비즈니스 규칙 패턴 (필수)

```python
# app/domain/admin/services/order_service.py
async def cancel_order(self, order_id: int):
    """비즈니스 규칙 적용 - 필수"""
    order = session.query(Order).filter(Order.id == order_id).first()

    # 배송 완료된 주문은 취소 불가
    if order.status == OrderStatus.DELIVERED:
        raise ValueError("배송 완료된 주문은 취소할 수 없습니다")

    order.status = OrderStatus.CANCELLED
    # ...

async def get_order(self, order_id: int):
    """관계 조인 - 필수"""
    order = (
        session.query(Order)
        .join(Consumer, Order.consumer_id == Consumer.id)
        .join(Product, Order.product_id == Product.id)
        .filter(Order.id == order_id)
        .first()
    )
    # 소비자/상품 정보 포함

async def list_orders(self, ..., consumer_id: Optional[int] = None, status: Optional[str] = None):
    """필터링 지원 - 필수"""
    query = session.query(Order)
    if consumer_id:
        query = query.filter(Order.consumer_id == consumer_id)
    if status:
        query = query.filter(Order.status == OrderStatus(status))
    # ...
```

## 4. Router use_policy 파라미터 패턴 (필수)

```python
# app/api/v1/admin/{entity}_router.py
@router.post("/create")
async def create_entity(entity: EntityCreateModel, use_policy: bool = False):
    """use_policy 파라미터 필수"""
    flow = EntityFlow()
    result = await flow.process_request(
        action="create",
        data=entity.model_dump(),
        use_policy=use_policy  # 필수
    )
    return result
```

## 5. 어댑터 경로 찾기 로직 (필수)

```python
def _load_adapter(self):
    """어댑터 자동 로드 - 필수 구현"""
    project_root = Path(__file__).parent.parent.parent.parent.parent
    adapter_base_path = project_root / "artifacts" / "fine-tuned-adapters" / "customer-service"

    if adapter_base_path.exists():
        lora_path = adapter_base_path / "customer_service" / "lora"
        if lora_path.exists():
            # 가장 최근 run_* 디렉토리 찾기
            run_dirs = sorted(
                [d for d in lora_path.iterdir()
                 if d.is_dir() and d.name.startswith("run_")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if run_dirs:
                self.adapter_path = run_dirs[0]
            else:
                # fixed_model 사용
                fixed_model = lora_path / "fixed_model"
                if fixed_model.exists():
                    self.adapter_path = fixed_model
```

