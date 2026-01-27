"""선수 데이터 Repository.

데이터베이스 접근 로직을 담당합니다.
"""
import logging
from typing import List, Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.domain.v10.soccer.models.bases.players import Player

logger = logging.getLogger(__name__)


class PlayerRepository:
    """선수 데이터 Repository.

    Neon 데이터베이스의 players 테이블에 대한 CRUD 작업을 수행합니다.
    """

    def __init__(self, session: AsyncSession):
        """PlayerRepository 초기화.

        Args:
            session: 데이터베이스 세션
        """
        self.session = session
        logger.debug("[Repository] PlayerRepository 초기화")

    async def find_by_id(self, player_id: int) -> Optional[Player]:
        """ID로 선수를 조회합니다.

        Args:
            player_id: 선수 ID

        Returns:
            Player 객체 또는 None
        """
        result = await self.session.execute(
            select(Player).where(Player.id == player_id)
        )
        return result.scalar_one_or_none()

    async def create(self, player_data: Dict[str, Any]) -> Player:
        """새 선수를 생성합니다.

        Args:
            player_data: 선수 데이터 딕셔너리

        Returns:
            생성된 Player 객체

        Raises:
            IntegrityError: 중복 키 또는 제약 조건 위반 시
        """
        new_player = Player(**player_data)
        self.session.add(new_player)
        logger.debug(f"[Repository] 선수 생성: ID {player_data.get('id')}")
        return new_player

    async def update(self, player: Player, player_data: Dict[str, Any]) -> Player:
        """기존 선수를 업데이트합니다.

        Args:
            player: 업데이트할 Player 객체
            player_data: 업데이트할 데이터 딕셔너리

        Returns:
            업데이트된 Player 객체
        """
        for key, value in player_data.items():
            if key != "id":  # ID는 업데이트하지 않음
                setattr(player, key, value)
        logger.debug(f"[Repository] 선수 업데이트: ID {player.id}")
        return player

    async def upsert_batch(
        self,
        players_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """여러 선수를 일괄 upsert (insert or update) 합니다.

        Args:
            players_data: 선수 데이터 리스트

        Returns:
            처리 결과 딕셔너리
            {
                "inserted_count": 삽입된 개수,
                "updated_count": 업데이트된 개수,
                "error_count": 오류 개수,
                "errors": 오류 상세 정보 리스트
            }
        """
        inserted_count = 0
        updated_count = 0
        error_count = 0
        errors = []

        for player_data in players_data:
            try:
                player_id = player_data.get("id")
                if not player_id:
                    error_msg = "ID가 없습니다"
                    logger.warning(f"[Repository] {error_msg}: {player_data}")
                    error_count += 1
                    errors.append({"item": player_data, "error": error_msg})
                    continue

                # 기존 선수 확인
                existing_player = await self.find_by_id(player_id)

                if existing_player:
                    # 업데이트
                    await self.update(existing_player, player_data)
                    updated_count += 1
                    logger.debug(f"[Repository] 선수 업데이트: ID {player_id}")
                else:
                    # 새로 삽입
                    await self.create(player_data)
                    inserted_count += 1
                    logger.debug(f"[Repository] 선수 삽입: ID {player_id}")

            except IntegrityError as e:
                error_count += 1
                error_msg = f"무결성 제약 조건 위반: {str(e)}"
                logger.error(f"[Repository] {error_msg}: ID {player_data.get('id')}", exc_info=True)
                errors.append({"item": player_data, "error": error_msg})
            except Exception as e:
                error_count += 1
                error_msg = f"처리 중 오류: {str(e)}"
                logger.error(
                    f"[Repository] {error_msg}: ID {player_data.get('id')}",
                    exc_info=True
                )
                errors.append({"item": player_data, "error": error_msg})

        return {
            "inserted_count": inserted_count,
            "updated_count": updated_count,
            "error_count": error_count,
            "errors": errors,
        }

    async def commit(self):
        """변경사항을 커밋합니다.

        Raises:
            Exception: 커밋 실패 시
        """
        try:
            await self.session.commit()
            logger.debug("[Repository] 커밋 완료")
        except Exception as e:
            await self.session.rollback()
            logger.error(f"[Repository] 커밋 실패, 롤백: {e}", exc_info=True)
            raise

