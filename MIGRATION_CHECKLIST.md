# 새 프로젝트 마이그레이션 필수 체크리스트

## 📋 필수 작업 목록

### 1. 프로젝트 구조 설정

```
app/
├── core/
│   └── config.py          # ✅ Neon DB 설정 (DATABASE_URL)
├── domain/
│   └── admin/
│       ├── bases/          # ✅ SQLAlchemy 모델 (Base 상속)
│       │   ├── products.py
│       │   ├── consumers.py
│       │   └── orders.py
│       ├── models/         # ✅ Pydantic 모델
│       │   ├── product_model.py
│       │   ├── consumer_model.py
│       │   └── order_model.py
│       ├── states/         # ✅ 상태 머신
│       │   ├── product_state.py
│       │   ├── consumer_state.py
│       │   └── order_state.py
│       ├── services/       # ✅ 규칙 기반 서비스
│       │   ├── product_service.py
│       │   ├── consumer_service.py
│       │   └── order_service.py
│       ├── agents/         # ✅ 정책 기반 에이전트
│       │   ├── base_agent.py
│       │   ├── product_agent.py
│       │   ├── consumer_agent.py
│       │   └── order_agent.py
│       └── orchestrators/  # ✅ 플로우 오케스트레이터
│           ├── product_flow.py
│           ├── consumer_flow.py
│           └── order_flow.py
├── api/
│   └── v1/
│       └── admin/
│           ├── product_router.py
│           ├── consumer_router.py
│           └── order_router.py
└── main.py                 # ✅ FastAPI 앱 설정
```

### 2. 공통 Base 모델 설정

**`app/domain/shared/bases/__init__.py`**
```python
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """공통 SQLAlchemy Base 클래스."""
    pass
```

### 3. 핵심 설정 파일

#### `app/core/config.py` - ✅ 필수
- Neon DB `DATABASE_URL` 환경변수 읽기
- `database_url` 프로퍼티로 연결 문자열 반환
- psycopg2 호환성 처리 (`channel_binding` 제거)

#### `app/main.py` - ✅ 필수
- 로깅 설정
- DB 연결 테스트 함수
- FastAPI 인스턴스 생성
- CORS 미들웨어
- 전역 예외 핸들러
- 라우터 등록

### 4. SQLAlchemy 모델 (Base 상속) - ✅ 필수

#### `app/domain/admin/bases/products.py`
- `Product` 모델
- `app.domain.shared.bases.Base` 상속

#### `app/domain/admin/bases/consumers.py`
- `Consumer` 모델
- `Order` 모델 (교차 엔티티)
- `OrderStatus` Enum
- 관계 설정: `Consumer.orders`, `Order.consumer`, `Order.product`

#### `app/domain/admin/bases/orders.py`
- `Order` 모델 (또는 consumers.py에 포함)
- 외래키: `consumer_id`, `product_id`
- CHECK 제약조건: `quantity > 0`, `price >= 0`

### 5. Pydantic 모델 - ✅ 필수

각 엔티티별로:
- `{Entity}Model` - 기본 전송 모델
- `{Entity}CreateModel` - 생성 요청 모델
- `{Entity}UpdateModel` - 수정 요청 모델
- `OrderDetailModel` - 관계 포함 상세 모델

### 6. 상태 머신 - ✅ 필수

#### `app/domain/admin/states/{entity}_state.py`
- `{Entity}Status` Enum
- `{Entity}State` 클래스 (Pydantic BaseModel 상속)
- 상태 전이 규칙 (`_valid_transitions`)
- `can_transition_to()` 메서드
- `transition_to()` 메서드
- 상태 이력 추적 (`status_history`)

### 7. 규칙 기반 서비스 - ✅ 필수

#### `app/domain/admin/services/{entity}_service.py`
- SQLAlchemy 세션 관리
- CRUD 작업 구현
- **비즈니스 규칙 적용**:
  - Order: 배송 완료 주문 삭제/취소 불가
  - Order: total_price 자동 계산
  - 존재 여부 검증 (consumer, product)
- **관계 조인**: 주문 조회 시 소비자/상품 정보 포함
- **필터링**: 목록 조회 시 consumer_id, status 필터 지원

### 8. 정책 기반 에이전트 - ✅ 필수

#### `app/domain/admin/agents/{entity}_agent.py`
- **BaseAgent 상속** 필수
- **Fine-tuned 어댑터 로드**:
  ```python
  def _load_model(self):
      # PeftModel.from_pretrained() 사용
      # artifacts/fine-tuned-adapters/customer-service 경로에서 로드
  ```
- **모델 미사용 시 폴백 처리**:
  ```python
  if self.model is None:
      return await self._fallback_execute(...)
  ```
- `execute()` 메서드 구현
- 프롬프트 생성 및 모델 추론
- 응답 파싱

### 9. 오케스트레이터 플로우 - ✅ 필수

#### `app/domain/admin/orchestrators/{entity}_flow.py`
- **어댑터 자동 로드**:
  ```python
  def _load_adapter(self):
      # artifacts/fine-tuned-adapters/customer-service 경로 찾기
      # 가장 최근 run_* 디렉토리 또는 fixed_model 사용
  ```
- **규칙/정책 분기**:
  ```python
  async def process_request(..., use_policy: bool = False):
      if use_policy:
          return await self._process_with_agent(...)
      else:
          return await self._process_with_service(...)
  ```
- `_process_with_service()` - 규칙 기반
- `_process_with_agent()` - 정책 기반

### 10. API 라우터 - ✅ 필수

#### `app/api/v1/admin/{entity}_router.py`
- FastAPI `APIRouter` 사용
- **`use_policy` 파라미터** 포함:
  ```python
  async def create_entity(..., use_policy: bool = False):
      flow = EntityFlow()
      result = await flow.process_request(
          action="create",
          data=...,
          use_policy=use_policy
      )
  ```
- CRUD 엔드포인트 구현

### 11. 의존성 패키지 - ✅ 필수

```txt
# requirements.txt
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.5
psycopg>=3.1.0
python-dotenv>=1.0.0

# 정책 기반 에이전트용 (선택)
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
bitsandbytes>=0.41.0  # GPU 양자화용
```

### 12. 환경 변수 설정 - ✅ 필수

**.env 파일**
```env
# Neon DB 연결
DATABASE_URL=postgresql://user:password@host:port/database

# 선택사항
DEBUG=false
OPENAI_API_KEY=your_key_here
```

### 13. 어댑터 디렉토리 구조 - ✅ 필수

```
artifacts/
└── fine-tuned-adapters/
    └── customer-service/
        └── customer_service/
            └── lora/
                ├── run_YYYYMMDD_HHMM/  # 가장 최근 실행
                │   ├── adapter_config.json
                │   └── adapter_model.safetensors
                └── fixed_model/        # 또는 이 경로 사용
                    ├── adapter_config.json
                    └── adapter_model.safetensors
```

### 14. 데이터베이스 DDL - ✅ 필수

- `products` 테이블
- `consumers` 테이블
- `orders` 테이블 (교차 엔티티)
- 외래키 제약조건
- CHECK 제약조건
- 인덱스
- 트리거 (updated_at 자동 갱신)

## 🔑 핵심 구현 포인트

### 1. 어댑터 자동 로드 로직
```python
def _load_adapter(self):
    project_root = Path(__file__).parent.parent.parent.parent.parent
    adapter_base_path = project_root / "artifacts" / "fine-tuned-adapters" / "customer-service"

    # 가장 최근 run_* 디렉토리 찾기
    run_dirs = sorted(
        [d for d in lora_path.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    if run_dirs:
        self.adapter_path = run_dirs[0]
```

### 2. 규칙/정책 분기 로직
```python
async def process_request(..., use_policy: bool = False):
    if use_policy:
        # 정책 기반 (Agent)
        return await self._process_with_agent(...)
    else:
        # 규칙 기반 (Service)
        return await self._process_with_service(...)
```

### 3. BaseAgent 상속 패턴
```python
class EntityAgent(BaseAgent):
    def __init__(self, adapter_path: Optional[Path] = None):
        super().__init__(name="EntityAgent", instruction="...")
        self.adapter_path = adapter_path
        self._load_model()

    async def execute(self, task: str, context: Dict[str, Any]):
        if self.model is None:
            return await self._fallback_execute(...)
        # 모델 사용 로직
```

### 4. 폴백 처리 패턴
```python
async def _fallback_execute(self, action: str, data: Dict, id: int):
    """모델이 없을 때 기본 처리."""
    return {
        "status": "success",
        "action": action,
        "message": "기본 처리 완료 (모델 미사용)",
        "data": data
    }
```

### 5. 비즈니스 규칙 적용
```python
# OrderService.cancel_order()
if order.status == OrderStatus.DELIVERED:
    raise ValueError("배송 완료된 주문은 취소할 수 없습니다")
```

### 6. 관계 조인 및 필터링
```python
# OrderService.get_order()
order = (
    session.query(Order)
    .join(Consumer, Order.consumer_id == Consumer.id)
    .join(Product, Order.product_id == Product.id)
    .filter(Order.id == order_id)
    .first()
)

# OrderService.list_orders()
if consumer_id:
    query = query.filter(Order.consumer_id == consumer_id)
if status:
    query = query.filter(Order.status == OrderStatus(status))
```

## ✅ 검증 체크리스트

- [ ] `app/domain/shared/bases/__init__.py`에 Base 클래스 정의
- [ ] 모든 SQLAlchemy 모델이 Base 상속
- [ ] 모든 Agent가 BaseAgent 상속
- [ ] 모든 Flow에서 어댑터 자동 로드 구현
- [ ] 모든 Flow에서 `use_policy` 분기 구현
- [ ] 모든 Service에 비즈니스 규칙 적용
- [ ] 모든 Agent에 폴백 처리 구현
- [ ] 모든 Router에 `use_policy` 파라미터 포함
- [ ] Order 조회 시 관계 조인 구현
- [ ] Order 목록에 필터링 구현
- [ ] `.env`에 `DATABASE_URL` 설정
- [ ] `artifacts/fine-tuned-adapters/customer-service` 경로 존재

## 🚀 빠른 시작 명령어

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
cp env.example .env
# .env 파일 편집하여 DATABASE_URL 설정

# 3. 데이터베이스 테이블 생성
psql $DATABASE_URL -f consumers_orders_ddl.sql

# 4. 서버 실행
python -m app.main
```

