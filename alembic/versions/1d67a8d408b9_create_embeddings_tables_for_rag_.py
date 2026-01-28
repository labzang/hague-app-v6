"""create embeddings tables for RAG semantic search

Revision ID: 1d67a8d408b9
Revises: 54d0fe2fa553
Create Date: 2026-01-28 10:23:30.127465

RAG 및 시맨틱 검색을 위한 선수 및 팀 임베딩 벡터 테이블 생성.
KoElectra 모델(768 차원)을 사용한 임베딩을 저장하며,
pgvector의 HNSW 인덱스를 사용하여 코사인 유사도 검색을 지원합니다.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '1d67a8d408b9'
down_revision: Union[str, None] = '54d0fe2fa553'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """임베딩 테이블 및 인덱스 생성."""
    # players_embeddings 테이블 생성
    op.create_table(
        'players_embeddings',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False, comment='임베딩 레코드 고유 식별자'),
        sa.Column('player_id', sa.BigInteger(), nullable=False, comment='선수 ID'),
        sa.Column('content', sa.Text(), nullable=False, comment='임베딩 생성에 사용된 원본 텍스트'),
        sa.Column('embedding', Vector(768), nullable=False, comment='768차원 임베딩 벡터 (KoElectra)'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False, comment='레코드 생성 시간'),
        sa.ForeignKeyConstraint(['player_id'], ['players.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        comment='선수 임베딩 벡터를 저장하는 테이블'
    )

    # teams_embeddings 테이블 생성
    op.create_table(
        'teams_embeddings',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False, comment='임베딩 레코드 고유 식별자'),
        sa.Column('team_id', sa.BigInteger(), nullable=False, comment='팀 ID'),
        sa.Column('content', sa.Text(), nullable=False, comment='임베딩 생성에 사용된 원본 텍스트'),
        sa.Column('embedding', Vector(768), nullable=False, comment='768차원 임베딩 벡터 (KoElectra)'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False, comment='레코드 생성 시간'),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        comment='팀 임베딩 벡터를 저장하는 테이블'
    )

    # players_embeddings에 HNSW 인덱스 생성 (코사인 유사도)
    op.execute(
        """
        CREATE INDEX idx_players_embeddings
        ON players_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )

    # teams_embeddings에 HNSW 인덱스 생성 (코사인 유사도)
    op.execute(
        """
        CREATE INDEX idx_teams_embeddings
        ON teams_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )

    # 외래 키 인덱스 생성 (선택사항이지만 성능 향상을 위해 권장)
    op.create_index(
        'idx_players_embeddings_player_id',
        'players_embeddings',
        ['player_id'],
        unique=False
    )

    op.create_index(
        'idx_teams_embeddings_team_id',
        'teams_embeddings',
        ['team_id'],
        unique=False
    )


def downgrade() -> None:
    """임베딩 테이블 및 인덱스 삭제."""
    # 인덱스 삭제
    op.drop_index('idx_teams_embeddings_team_id', table_name='teams_embeddings')
    op.drop_index('idx_players_embeddings_player_id', table_name='players_embeddings')

    # HNSW 인덱스 삭제
    op.execute('DROP INDEX IF EXISTS idx_teams_embeddings;')
    op.execute('DROP INDEX IF EXISTS idx_players_embeddings;')

    # 테이블 삭제
    op.drop_table('teams_embeddings')
    op.drop_table('players_embeddings')

