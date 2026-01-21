-- Consumer와 Order 테이블 DDL
-- PostgreSQL 데이터베이스용

-- 1. 주문 상태 열거형 타입 생성
CREATE TYPE order_status AS ENUM (
    'pending',      -- 대기 중
    'confirmed',    -- 확인됨
    'processing',   -- 처리 중
    'shipped',      -- 배송 중
    'delivered',    -- 배송 완료
    'cancelled'     -- 취소됨
);

-- 2. 소비자(Consumer) 테이블
CREATE TABLE IF NOT EXISTS consumers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    phone VARCHAR(20),
    address TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- 인덱스
    CONSTRAINT uk_consumers_email UNIQUE (email)
);

CREATE INDEX IF NOT EXISTS idx_consumers_email ON consumers(email);

-- 3. 주문(Order) 테이블 (교차 엔티티)
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    consumer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price INTEGER NOT NULL,
    total_price INTEGER NOT NULL,
    status order_status NOT NULL DEFAULT 'pending',
    order_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- 외래키 제약조건
    CONSTRAINT fk_orders_consumer
        FOREIGN KEY (consumer_id)
        REFERENCES consumers(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,

    CONSTRAINT fk_orders_product
        FOREIGN KEY (product_id)
        REFERENCES products(id)
        ON DELETE RESTRICT
        ON UPDATE CASCADE,

    -- CHECK 제약조건
    CONSTRAINT check_quantity_positive
        CHECK (quantity > 0),

    CONSTRAINT check_unit_price_non_negative
        CHECK (unit_price >= 0),

    CONSTRAINT check_total_price_non_negative
        CHECK (total_price >= 0)
);

-- 인덱스 생성 (PostgreSQL 구문)
CREATE INDEX IF NOT EXISTS idx_orders_consumer_id ON orders(consumer_id);
CREATE INDEX IF NOT EXISTS idx_orders_product_id ON orders(product_id);
CREATE INDEX IF NOT EXISTS idx_orders_order_date ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

-- 복합 인덱스 (자주 함께 조회되는 경우)
CREATE INDEX IF NOT EXISTS idx_orders_consumer_date ON orders(consumer_id, order_date);
CREATE INDEX IF NOT EXISTS idx_orders_product_status ON orders(product_id, status);

-- 4. updated_at 자동 업데이트 트리거 함수
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 5. 트리거 생성
CREATE TRIGGER update_consumers_updated_at
    BEFORE UPDATE ON consumers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at
    BEFORE UPDATE ON orders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 6. 코멘트 추가
COMMENT ON TABLE consumers IS '소비자 정보를 저장하는 테이블';
COMMENT ON TABLE orders IS '주문 정보를 저장하는 테이블 (Product와 Consumer를 연결하는 교차 엔티티)';
COMMENT ON TYPE order_status IS '주문 상태 열거형';

COMMENT ON COLUMN consumers.id IS '소비자 고유 식별자';
COMMENT ON COLUMN consumers.name IS '소비자 이름';
COMMENT ON COLUMN consumers.email IS '이메일 주소';
COMMENT ON COLUMN consumers.phone IS '전화번호';
COMMENT ON COLUMN consumers.address IS '배송 주소';
COMMENT ON COLUMN consumers.created_at IS '생성 일시';
COMMENT ON COLUMN consumers.updated_at IS '수정 일시';

COMMENT ON COLUMN orders.id IS '주문 고유 식별자';
COMMENT ON COLUMN orders.consumer_id IS '소비자 ID (외래키)';
COMMENT ON COLUMN orders.product_id IS '상품 ID (외래키)';
COMMENT ON COLUMN orders.quantity IS '주문 수량';
COMMENT ON COLUMN orders.unit_price IS '단가 (주문 시점의 가격)';
COMMENT ON COLUMN orders.total_price IS '총 가격 (quantity * unit_price)';
COMMENT ON COLUMN orders.status IS '주문 상태';
COMMENT ON COLUMN orders.order_date IS '주문 일시';
COMMENT ON COLUMN orders.created_at IS '생성 일시';
COMMENT ON COLUMN orders.updated_at IS '수정 일시';

-- 7. 뷰: 주문 상세 정보 조인 뷰
CREATE OR REPLACE VIEW v_order_details AS
SELECT
    o.id AS order_id,
    o.quantity,
    o.unit_price,
    o.total_price,
    o.status,
    o.order_date,
    o.created_at AS order_created_at,
    o.updated_at AS order_updated_at,
    c.id AS consumer_id,
    c.name AS consumer_name,
    c.email AS consumer_email,
    c.phone AS consumer_phone,
    c.address AS consumer_address,
    p.id AS product_id,
    p.name AS product_name,
    p.description AS product_description,
    p.price AS product_current_price,
    p.category AS product_category,
    p.brand AS product_brand
FROM
    orders o
    INNER JOIN consumers c ON o.consumer_id = c.id
    INNER JOIN products p ON o.product_id = p.id;

COMMENT ON VIEW v_order_details IS '주문, 소비자, 상품 정보를 조인한 상세 뷰';

