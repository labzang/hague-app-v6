"""경기 일정 데이터 규칙 기반 서비스."""
import logging
from typing import List, Dict, Any

from app.core.database import AsyncSessionLocal
from app.domain.v10.soccer.hub.repositories.schedule_repository import ScheduleRepository

logger = logging.getLogger(__name__)


class ScheduleService:
    """경기 일정 데이터를 규칙 기반으로 처리하는 서비스.

    JSONL 데이터를 schedules 테이블에 삽입하는 규칙 기반 처리를 수행합니다.
    """

    def __init__(self):
        """ScheduleService 초기화."""
        logger.info("[서비스] ScheduleService 초기화")

    def _normalize_schedule_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """경기 일정 데이터를 정규화합니다.

        Args:
            item: 원본 경기 일정 데이터

        Returns:
            정규화된 경기 일정 데이터
        """
        normalized = {}

        # 필드 매핑 및 타입 변환
        if "id" in item:
            normalized["id"] = int(item["id"]) if item["id"] is not None else None

        if "stadium_id" in item:
            normalized["stadium_id"] = int(item["stadium_id"]) if item["stadium_id"] is not None else None

        if "hometeam_id" in item:
            normalized["hometeam_id"] = int(item["hometeam_id"]) if item["hometeam_id"] is not None else None

        if "awayteam_id" in item:
            normalized["awayteam_id"] = int(item["awayteam_id"]) if item["awayteam_id"] is not None else None

        if "stadium_code" in item:
            normalized["stadium_code"] = str(item["stadium_code"])[:10] if item["stadium_code"] else None

        if "sche_date" in item:
            normalized["sche_date"] = str(item["sche_date"])[:10] if item["sche_date"] else None

        if "gubun" in item:
            normalized["gubun"] = str(item["gubun"])[:10] if item["gubun"] else None

        if "hometeam_code" in item:
            normalized["hometeam_code"] = str(item["hometeam_code"])[:10] if item["hometeam_code"] else None

        if "awayteam_code" in item:
            normalized["awayteam_code"] = str(item["awayteam_code"])[:10] if item["awayteam_code"] else None

        if "home_score" in item:
            normalized["home_score"] = int(item["home_score"]) if item["home_score"] is not None else None

        if "away_score" in item:
            normalized["away_score"] = int(item["away_score"]) if item["away_score"] is not None else None

        return normalized

    async def _save_schedules_to_database(
        self,
        normalized_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """정규화된 경기 일정 데이터를 Repository를 통해 데이터베이스에 저장합니다.

        Args:
            normalized_items: 정규화된 경기 일정 데이터 리스트

        Returns:
            저장 결과 딕셔너리
        """
        async with AsyncSessionLocal() as session:
            repository = ScheduleRepository(session)
            logger.info("[서비스] Repository를 통해 데이터베이스 저장 시작...")
            db_result = await repository.upsert_batch(normalized_items)
            await repository.commit()
            logger.info(
                f"[서비스] 데이터베이스 저장 완료: "
                f"삽입 {db_result['inserted_count']}개, "
                f"업데이트 {db_result['updated_count']}개, "
                f"오류 {db_result['error_count']}개"
            )
        return db_result

    async def process_schedules(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """경기 일정 데이터를 규칙 기반으로 처리하고 데이터베이스에 삽입합니다.

        Args:
            items: 처리할 경기 일정 데이터 리스트

        Returns:
            처리 결과 딕셔너리
        """
        logger.info(f"[서비스] 규칙 기반 처리 시작: {len(items)}개 항목")

        # 1. 데이터 정규화
        logger.info("[서비스] 데이터 정규화 시작...")
        normalized_items = []
        for item in items:
            try:
                normalized = self._normalize_schedule_data(item)
                normalized_items.append(normalized)
            except Exception as e:
                logger.error(f"[서비스] 데이터 정규화 실패: {item.get('id', 'unknown')} - {e}", exc_info=True)

        logger.info(f"[서비스] 정규화 완료: {len(normalized_items)}개 항목")

        # 2. Repository를 통해 데이터베이스에 저장
        logger.info("[서비스] Repository를 통해 데이터베이스 저장 시작...")
        db_result = await self._save_schedules_to_database(normalized_items)

        result = {
            "success": True,
            "method": "rule_based",
            "total_items": len(items),
            "normalized_count": len(normalized_items),
            "database_result": db_result,
        }

        logger.info(
            f"[서비스] 규칙 기반 처리 완료: "
            f"총 {len(items)}개, 삽입 {db_result['inserted_count']}개, "
            f"업데이트 {db_result['updated_count']}개, 오류 {db_result['error_count']}개"
        )
        return result

