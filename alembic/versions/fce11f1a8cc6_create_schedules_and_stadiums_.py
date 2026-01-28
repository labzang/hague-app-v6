"""create schedules and stadiums embeddings tables

Revision ID: fce11f1a8cc6
Revises: 1d67a8d408b9
Create Date: 2026-01-28 10:38:02.282625

RAG 및 시맨틱 검색을 위한 경기 일정 및 경기장 임베딩 벡터 테이블 생성.
KoElectra 모델(768 차원)을 사용한 임베딩을 저장하며,
pgvector의 HNSW 인덱스를 사용하여 코사인 유사도 검색을 지원합니다.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'fce11f1a8cc6'
down_revision: Union[str, None] = '1d67a8d408b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """경기 일정 및 경기장 임베딩 테이블 및 인덱스 생성."""
    # schedules_embeddings 테이블 생성
    op.create_table(
        'schedules_embeddings',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False, comment='임베딩 레코드 고유 식별자'),
        sa.Column('schedule_id', sa.BigInteger(), nullable=False, comment='경기 일정 ID'),
        sa.Column('content', sa.Text(), nullable=False, comment='임베딩 생성에 사용된 원본 텍스트'),
        sa.Column('embedding', Vector(768), nullable=False, comment='768차원 임베딩 벡터 (KoElectra)'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False, comment='레코드 생성 시간'),
        sa.ForeignKeyConstraint(['schedule_id'], ['schedules.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        comment='경기 일정 임베딩 벡터를 저장하는 테이블'
    )

    # stadiums_embeddings 테이블 생성
    op.create_table(
        'stadiums_embeddings',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False, comment='임베딩 레코드 고유 식별자'),
        sa.Column('stadium_id', sa.BigInteger(), nullable=False, comment='경기장 ID'),
        sa.Column('content', sa.Text(), nullable=False, comment='임베딩 생성에 사용된 원본 텍스트'),
        sa.Column('embedding', Vector(768), nullable=False, comment='768차원 임베딩 벡터 (KoElectra)'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False, comment='레코드 생성 시간'),
        sa.ForeignKeyConstraint(['stadium_id'], ['stadiums.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        comment='경기장 임베딩 벡터를 저장하는 테이블'
    )

    # schedules_embeddings에 HNSW 인덱스 생성 (코사인 유사도)
    op.execute(
        """
        CREATE INDEX idx_schedules_embeddings
        ON schedules_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )

    # stadiums_embeddings에 HNSW 인덱스 생성 (코사인 유사도)
    op.execute(
        """
        CREATE INDEX idx_stadiums_embeddings
        ON stadiums_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )

    # 외래 키 인덱스 생성 (선택사항이지만 성능 향상을 위해 권장)
    op.create_index(
        'idx_schedules_embeddings_schedule_id',
        'schedules_embeddings',
        ['schedule_id'],
        unique=False
    )

    op.create_index(
        'idx_stadiums_embeddings_stadium_id',
        'stadiums_embeddings',
        ['stadium_id'],
        unique=False
    )


def downgrade() -> None:
    """경기 일정 및 경기장 임베딩 테이블 및 인덱스 삭제."""
    # 인덱스 삭제
    op.drop_index('idx_stadiums_embeddings_stadium_id', table_name='stadiums_embeddings')
    op.drop_index('idx_schedules_embeddings_schedule_id', table_name='schedules_embeddings')

    # HNSW 인덱스 삭제
    op.execute('DROP INDEX IF EXISTS idx_stadiums_embeddings;')
    op.execute('DROP INDEX IF EXISTS idx_schedules_embeddings;')

    # 테이블 삭제
    op.drop_table('stadiums_embeddings')
    op.drop_table('schedules_embeddings')

