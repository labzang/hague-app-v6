"""경기 일정 데이터 Repository.

데이터베이스 접근 로직을 담당합니다.
"""
import logging
from typing import List, Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.domain.v10.soccer.models.bases.schedules import Schedule

logger = logging.getLogger(__name__)


class ScheduleRepository:
    """경기 일정 데이터 Repository.

    Neon 데이터베이스의 schedules 테이블에 대한 CRUD 작업을 수행합니다.
    """

    def __init__(self, session: AsyncSession):
        """ScheduleRepository 초기화.

        Args:
            session: 데이터베이스 세션
        """
        self.session = session
        logger.debug("[Repository] ScheduleRepository 초기화")

    async def find_by_id(self, schedule_id: int) -> Optional[Schedule]:
        """ID로 경기 일정을 조회합니다.

        Args:
            schedule_id: 경기 일정 ID

        Returns:
            Schedule 객체 또는 None
        """
        result = await self.session.execute(
            select(Schedule).where(Schedule.id == schedule_id)
        )
        return result.scalar_one_or_none()

    async def create(self, schedule_data: Dict[str, Any]) -> Schedule:
        """새 경기 일정을 생성합니다.

        Args:
            schedule_data: 경기 일정 데이터 딕셔너리

        Returns:
            생성된 Schedule 객체

        Raises:
            IntegrityError: 중복 키 또는 제약 조건 위반 시
        """
        new_schedule = Schedule(**schedule_data)
        self.session.add(new_schedule)
        logger.debug(f"[Repository] 경기 일정 생성: ID {schedule_data.get('id')}")
        return new_schedule

    async def update(self, schedule: Schedule, schedule_data: Dict[str, Any]) -> Schedule:
        """기존 경기 일정을 업데이트합니다.

        Args:
            schedule: 업데이트할 Schedule 객체
            schedule_data: 업데이트할 데이터 딕셔너리

        Returns:
            업데이트된 Schedule 객체
        """
        for key, value in schedule_data.items():
            if key != "id":  # ID는 업데이트하지 않음
                setattr(schedule, key, value)
        logger.debug(f"[Repository] 경기 일정 업데이트: ID {schedule.id}")
        return schedule

    async def upsert_batch(
        self,
        schedules_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """여러 경기 일정을 일괄 upsert (insert or update) 합니다.

        Args:
            schedules_data: 경기 일정 데이터 리스트

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

        for schedule_data in schedules_data:
            try:
                schedule_id = schedule_data.get("id")
                if not schedule_id:
                    error_msg = "ID가 없습니다"
                    logger.warning(f"[Repository] {error_msg}: {schedule_data}")
                    error_count += 1
                    errors.append({"item": schedule_data, "error": error_msg})
                    continue

                # 기존 경기 일정 확인
                existing_schedule = await self.find_by_id(schedule_id)

                if existing_schedule:
                    # 업데이트
                    await self.update(existing_schedule, schedule_data)
                    updated_count += 1
                    logger.debug(f"[Repository] 경기 일정 업데이트: ID {schedule_id}")
                else:
                    # 새로 삽입
                    await self.create(schedule_data)
                    inserted_count += 1
                    logger.debug(f"[Repository] 경기 일정 삽입: ID {schedule_id}")

            except IntegrityError as e:
                error_count += 1
                error_msg = f"무결성 제약 조건 위반: {str(e)}"
                logger.error(f"[Repository] {error_msg}: ID {schedule_data.get('id')}", exc_info=True)
                errors.append({"item": schedule_data, "error": error_msg})
            except Exception as e:
                error_count += 1
                error_msg = f"처리 중 오류: {str(e)}"
                logger.error(
                    f"[Repository] {error_msg}: ID {schedule_data.get('id')}",
                    exc_info=True
                )
                errors.append({"item": schedule_data, "error": error_msg})

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

