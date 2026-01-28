"""경기장 데이터 Repository.

데이터베이스 접근 로직을 담당합니다.
"""
import logging
from typing import List, Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.domain.v10.soccer.models.bases.stadiums import Stadium

logger = logging.getLogger(__name__)


class StadiumRepository:
    """경기장 데이터 Repository.

    Neon 데이터베이스의 stadiums 테이블에 대한 CRUD 작업을 수행합니다.
    """

    def __init__(self, session: AsyncSession):
        """StadiumRepository 초기화.

        Args:
            session: 데이터베이스 세션
        """
        self.session = session
        logger.debug("[Repository] StadiumRepository 초기화")

    async def find_by_id(self, stadium_id: int) -> Optional[Stadium]:
        """ID로 경기장을 조회합니다.

        Args:
            stadium_id: 경기장 ID

        Returns:
            Stadium 객체 또는 None
        """
        result = await self.session.execute(
            select(Stadium).where(Stadium.id == stadium_id)
        )
        return result.scalar_one_or_none()

    async def create(self, stadium_data: Dict[str, Any]) -> Stadium:
        """새 경기장을 생성합니다.

        Args:
            stadium_data: 경기장 데이터 딕셔너리

        Returns:
            생성된 Stadium 객체

        Raises:
            IntegrityError: 중복 키 또는 제약 조건 위반 시
        """
        new_stadium = Stadium(**stadium_data)
        self.session.add(new_stadium)
        logger.debug(f"[Repository] 경기장 생성: ID {stadium_data.get('id')}")
        return new_stadium

    async def update(self, stadium: Stadium, stadium_data: Dict[str, Any]) -> Stadium:
        """기존 경기장을 업데이트합니다.

        Args:
            stadium: 업데이트할 Stadium 객체
            stadium_data: 업데이트할 데이터 딕셔너리

        Returns:
            업데이트된 Stadium 객체
        """
        for key, value in stadium_data.items():
            if key != "id":  # ID는 업데이트하지 않음
                setattr(stadium, key, value)
        logger.debug(f"[Repository] 경기장 업데이트: ID {stadium.id}")
        return stadium

    async def upsert_batch(
        self,
        stadiums_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """여러 경기장을 일괄 upsert (insert or update) 합니다.

        Args:
            stadiums_data: 경기장 데이터 리스트

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

        for stadium_data in stadiums_data:
            try:
                stadium_id = stadium_data.get("id")
                if not stadium_id:
                    error_msg = "ID가 없습니다"
                    logger.warning(f"[Repository] {error_msg}: {stadium_data}")
                    error_count += 1
                    errors.append({"item": stadium_data, "error": error_msg})
                    continue

                # 기존 경기장 확인
                existing_stadium = await self.find_by_id(stadium_id)

                if existing_stadium:
                    # 업데이트
                    await self.update(existing_stadium, stadium_data)
                    updated_count += 1
                    logger.debug(f"[Repository] 경기장 업데이트: ID {stadium_id}")
                else:
                    # 새로 삽입
                    await self.create(stadium_data)
                    inserted_count += 1
                    logger.debug(f"[Repository] 경기장 삽입: ID {stadium_id}")

            except IntegrityError as e:
                error_count += 1
                error_msg = f"무결성 제약 조건 위반: {str(e)}"
                logger.error(f"[Repository] {error_msg}: ID {stadium_data.get('id')}", exc_info=True)
                errors.append({"item": stadium_data, "error": error_msg})
            except Exception as e:
                error_count += 1
                error_msg = f"처리 중 오류: {str(e)}"
                logger.error(
                    f"[Repository] {error_msg}: ID {stadium_data.get('id')}",
                    exc_info=True
                )
                errors.append({"item": stadium_data, "error": error_msg})

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

