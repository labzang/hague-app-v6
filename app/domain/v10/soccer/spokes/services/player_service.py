"""선수 데이터 규칙 기반 서비스."""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.core.database import AsyncSessionLocal
from app.domain.v10.soccer.hub.repositories.player_repository import PlayerRepository

logger = logging.getLogger(__name__)


class PlayerService:
    """선수 데이터를 규칙 기반으로 처리하는 서비스.

    JSONL 데이터를 players 테이블에 삽입하는 규칙 기반 처리를 수행합니다.
    """

    def __init__(self):
        """PlayerService 초기화."""
        logger.info("[서비스] PlayerService 초기화")

    def _normalize_player_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """선수 데이터를 정규화합니다.

        Args:
            item: 원본 선수 데이터

        Returns:
            정규화된 선수 데이터
        """
        normalized = {}

        # 필드 매핑 및 타입 변환
        if "id" in item:
            normalized["id"] = int(item["id"]) if item["id"] is not None else None

        if "team_id" in item:
            normalized["team_id"] = int(item["team_id"]) if item["team_id"] is not None else None

        if "player_name" in item:
            normalized["player_name"] = str(item["player_name"])[:20] if item["player_name"] else None

        if "e_player_name" in item:
            normalized["e_player_name"] = str(item["e_player_name"])[:40] if item["e_player_name"] else None

        if "nickname" in item:
            normalized["nickname"] = str(item["nickname"])[:30] if item["nickname"] else None

        if "join_yyyy" in item:
            normalized["join_yyyy"] = str(item["join_yyyy"])[:10] if item["join_yyyy"] else None

        if "position" in item:
            normalized["position"] = str(item["position"])[:10] if item["position"] else None

        if "back_no" in item:
            normalized["back_no"] = int(item["back_no"]) if item["back_no"] is not None else None

        if "nation" in item:
            normalized["nation"] = str(item["nation"])[:20] if item["nation"] else None

        if "birth_date" in item:
            birth_date = item["birth_date"]
            if birth_date:
                try:
                    # 문자열을 날짜로 변환
                    if isinstance(birth_date, str):
                        normalized["birth_date"] = datetime.strptime(birth_date, "%Y-%m-%d").date()
                    else:
                        normalized["birth_date"] = birth_date
                except (ValueError, TypeError):
                    logger.warning(f"[서비스] 생년월일 파싱 실패: {birth_date}")
                    normalized["birth_date"] = None
            else:
                normalized["birth_date"] = None

        if "solar" in item:
            normalized["solar"] = str(item["solar"])[:10] if item["solar"] else None

        if "height" in item:
            normalized["height"] = int(item["height"]) if item["height"] is not None else None

        if "weight" in item:
            normalized["weight"] = int(item["weight"]) if item["weight"] is not None else None

        return normalized

    async def _save_players_to_database(
        self,
        normalized_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """정규화된 선수 데이터를 Repository를 통해 데이터베이스에 저장합니다.

        Args:
            normalized_items: 정규화된 선수 데이터 리스트

        Returns:
            저장 결과 딕셔너리
        """
        async with AsyncSessionLocal() as session:
            # Repository 인스턴스 생성
            repository = PlayerRepository(session)

            # 일괄 upsert 수행
            logger.info("[서비스] Repository를 통해 데이터베이스 저장 시작...")
            db_result = await repository.upsert_batch(normalized_items)

            # 커밋
            await repository.commit()
            logger.info(
                f"[서비스] 데이터베이스 저장 완료: "
                f"삽입 {db_result['inserted_count']}개, "
                f"업데이트 {db_result['updated_count']}개, "
                f"오류 {db_result['error_count']}개"
            )

        return db_result

    async def process_players(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """선수 데이터를 규칙 기반으로 처리하고 데이터베이스에 삽입합니다.

        Args:
            items: 처리할 선수 데이터 리스트

        Returns:
            처리 결과 딕셔너리
        """
        logger.info(f"[서비스] 규칙 기반 처리 시작: {len(items)}개 항목")

        # 1. 데이터 정규화
        logger.info("[서비스] 데이터 정규화 시작...")
        normalized_items = []
        for item in items:
            try:
                normalized = self._normalize_player_data(item)
                normalized_items.append(normalized)
            except Exception as e:
                logger.error(f"[서비스] 데이터 정규화 실패: {item.get('id', 'unknown')} - {e}", exc_info=True)

        logger.info(f"[서비스] 정규화 완료: {len(normalized_items)}개 항목")

        # 2. Repository를 통해 데이터베이스에 저장
        logger.info("[서비스] Repository를 통해 데이터베이스 저장 시작...")
        db_result = await self._save_players_to_database(normalized_items)

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

