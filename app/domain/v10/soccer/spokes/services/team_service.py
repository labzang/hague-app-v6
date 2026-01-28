"""팀 데이터 규칙 기반 서비스."""
import logging
from typing import List, Dict, Any

from app.core.database import AsyncSessionLocal
from app.domain.v10.soccer.hub.repositories.team_repository import TeamRepository

logger = logging.getLogger(__name__)


class TeamService:
    """팀 데이터를 규칙 기반으로 처리하는 서비스.

    JSONL 데이터를 teams 테이블에 삽입하는 규칙 기반 처리를 수행합니다.
    """

    def __init__(self):
        """TeamService 초기화."""
        logger.info("[서비스] TeamService 초기화")

    def _normalize_team_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """팀 데이터를 정규화합니다.

        Args:
            item: 원본 팀 데이터

        Returns:
            정규화된 팀 데이터
        """
        normalized = {}

        # 필드 매핑 및 타입 변환
        if "id" in item:
            normalized["id"] = int(item["id"]) if item["id"] is not None else None

        if "stadium_id" in item:
            normalized["stadium_id"] = int(item["stadium_id"]) if item["stadium_id"] is not None else None

        if "team_code" in item:
            normalized["team_code"] = str(item["team_code"])[:10] if item["team_code"] else None

        if "region_name" in item:
            normalized["region_name"] = str(item["region_name"])[:10] if item["region_name"] else None

        if "team_name" in item:
            normalized["team_name"] = str(item["team_name"])[:40] if item["team_name"] else None

        if "e_team_name" in item:
            normalized["e_team_name"] = str(item["e_team_name"])[:50] if item["e_team_name"] else None

        if "orig_yyyy" in item:
            normalized["orig_yyyy"] = str(item["orig_yyyy"])[:10] if item["orig_yyyy"] else None

        if "zip_code1" in item:
            normalized["zip_code1"] = str(item["zip_code1"])[:10] if item["zip_code1"] else None

        if "zip_code2" in item:
            normalized["zip_code2"] = str(item["zip_code2"])[:10] if item["zip_code2"] else None

        if "address" in item:
            normalized["address"] = str(item["address"])[:80] if item["address"] else None

        if "ddd" in item:
            normalized["ddd"] = str(item["ddd"])[:10] if item["ddd"] else None

        if "tel" in item:
            normalized["tel"] = str(item["tel"])[:20] if item["tel"] else None

        if "fax" in item:
            normalized["fax"] = str(item["fax"])[:20] if item["fax"] else None

        if "homepage" in item:
            normalized["homepage"] = str(item["homepage"])[:100] if item["homepage"] else None

        if "owner" in item:
            normalized["owner"] = str(item["owner"])[:50] if item["owner"] else None

        return normalized

    async def _save_teams_to_database(
        self,
        normalized_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """정규화된 팀 데이터를 Repository를 통해 데이터베이스에 저장합니다.

        Args:
            normalized_items: 정규화된 팀 데이터 리스트

        Returns:
            저장 결과 딕셔너리
        """
        async with AsyncSessionLocal() as session:
            repository = TeamRepository(session)
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

    async def process_teams(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """팀 데이터를 규칙 기반으로 처리하고 데이터베이스에 삽입합니다.

        Args:
            items: 처리할 팀 데이터 리스트

        Returns:
            처리 결과 딕셔너리
        """
        logger.info(f"[서비스] 규칙 기반 처리 시작: {len(items)}개 항목")

        # 1. 데이터 정규화
        logger.info("[서비스] 데이터 정규화 시작...")
        normalized_items = []
        for item in items:
            try:
                normalized = self._normalize_team_data(item)
                normalized_items.append(normalized)
            except Exception as e:
                logger.error(f"[서비스] 데이터 정규화 실패: {item.get('id', 'unknown')} - {e}", exc_info=True)

        logger.info(f"[서비스] 정규화 완료: {len(normalized_items)}개 항목")

        # 2. Repository를 통해 데이터베이스에 저장
        logger.info("[서비스] Repository를 통해 데이터베이스 저장 시작...")
        db_result = await self._save_teams_to_database(normalized_items)

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

