"""선수 데이터 처리 오케스트레이터.

GoF 전략 패턴을 사용하여 정책기반/규칙기반 처리를 분기합니다.
"""
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers가 설치되지 않았습니다.")

from app.domain.v10.soccer.spokes.agents.player_agent import PlayerAgent
from app.domain.v10.soccer.spokes.services.player_service import PlayerService

logger = logging.getLogger(__name__)


class PlayerProcessingStrategy(ABC):
    """선수 데이터 처리 전략 인터페이스."""

    @abstractmethod
    async def process(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """선수 데이터를 처리합니다.

        Args:
            items: 처리할 선수 데이터 리스트

        Returns:
            처리 결과 딕셔너리
        """
        pass


class PolicyBasedStrategy(PlayerProcessingStrategy):
    """정책 기반 처리 전략.

    PlayerAgent를 사용하여 정책 기반 처리를 수행합니다.
    """

    def __init__(self):
        """PolicyBasedStrategy 초기화."""
        self.agent = PlayerAgent()
        logger.info("[전략] 정책 기반 전략 초기화 완료")

    async def process(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """정책 기반으로 선수 데이터를 처리합니다.

        Args:
            items: 처리할 선수 데이터 리스트

        Returns:
            처리 결과 딕셔너리
        """
        logger.info(f"[정책 기반] {len(items)}개 항목 처리 시작")
        result = await self.agent.process_players(items)
        logger.info("[정책 기반] 처리 완료")
        return result


class RuleBasedStrategy(PlayerProcessingStrategy):
    """규칙 기반 처리 전략.

    PlayerService를 사용하여 규칙 기반 처리를 수행합니다.
    """

    def __init__(self):
        """RuleBasedStrategy 초기화."""
        self.service = PlayerService()
        logger.info("[전략] 규칙 기반 전략 초기화 완료")

    async def process(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """규칙 기반으로 선수 데이터를 처리합니다.

        Args:
            items: 처리할 선수 데이터 리스트

        Returns:
            처리 결과 딕셔너리
        """
        logger.info(f"[규칙 기반] {len(items)}개 항목 처리 시작")
        result = await self.service.process_players(items)
        logger.info("[규칙 기반] 처리 완료")
        return result


class PlayerOrchestrator:
    """선수 데이터 처리 오케스트레이터.

    KoELECTRA 모델을 사용하여 정책기반/규칙기반을 판단하고
    적절한 전략을 선택하여 처리합니다.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
    ):
        """PlayerOrchestrator 초기화.

        Args:
            model_dir: KoELECTRA 모델 디렉토리 경로
        """
        self.model = None
        self.tokenizer = None
        self.model_dir = model_dir or self._get_default_model_dir()

        # 전략 인스턴스 생성
        self.policy_strategy = PolicyBasedStrategy()
        self.rule_strategy = RuleBasedStrategy()

        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.warning("[오케스트레이터] transformers 미설치, 기본 규칙 기반 사용")

    def _get_default_model_dir(self) -> Path:
        """기본 모델 디렉토리 경로를 반환합니다.

        Returns:
            모델 디렉토리 Path
        """
        # 프로젝트 루트 기준으로 artifacts 폴더 찾기
        current_file = Path(__file__)
        # app/domain/v10/soccer/hub/orchestrators/player_orchestrator.py
        # -> artifacts/models--monologg--koelectra-small-v3-discriminator
        project_root = current_file.parent.parent.parent.parent.parent.parent
        model_dir = project_root / "artifacts" / "models--monologg--koelectra-small-v3-discriminator"
        return model_dir

    def _load_model(self):
        """KoELECTRA 모델과 토크나이저를 로드합니다."""
        if not self.model_dir.exists():
            logger.warning(
                f"[오케스트레이터] 모델 디렉토리를 찾을 수 없습니다: {self.model_dir}. "
                "기본 규칙 기반 사용"
            )
            return

        try:
            logger.info(f"[오케스트레이터] 모델 로딩 시작: {self.model_dir}")

            # snapshots 폴더에서 최신 스냅샷 찾기
            snapshots_dir = self.model_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    # 가장 최근 스냅샷 사용 (일반적으로 해시 이름)
                    latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                    model_path = latest_snapshot
                    logger.info(f"[오케스트레이터] 스냅샷 사용: {model_path}")
                else:
                    model_path = self.model_dir
            else:
                model_path = self.model_dir

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
            )

            # 모델 로드 (SequenceClassification용)
            # discriminator 모델이므로 분류 작업에 사용
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                )
            except Exception:
                # SequenceClassification이 없으면 일반 모델 사용
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                )

            # 평가 모드로 설정
            self.model.eval()

            logger.info("[오케스트레이터] 모델 로딩 완료")

        except Exception as e:
            logger.error(f"[오케스트레이터] 모델 로딩 실패: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None

    def _determine_strategy_type(self, items: List[Dict[str, Any]]) -> str:
        """선수 데이터를 분석하여 정책기반/규칙기반을 판단합니다.

        휴리스틱 기반 판단:
        - 데이터베이스 삽입 작업은 항상 규칙 기반으로 처리
        - 복잡한 데이터 변환이나 정책 결정이 필요한 경우만 정책 기반

        Args:
            items: 선수 데이터 리스트

        Returns:
            "policy" 또는 "rule"
        """
        # 휴리스틱 1: 데이터베이스 삽입 작업은 항상 규칙 기반
        # JSONL 데이터를 players 테이블에 추가하는 것은 규칙 기반 처리
        logger.info("[판단] 데이터베이스 삽입 작업은 규칙 기반으로 처리")
        return "rule"

        # 향후 확장: 복잡한 데이터 변환이나 정책 결정이 필요한 경우
        # 아래 로직을 활성화하여 정책 기반으로 처리 가능
        # try:
        #     # 데이터 품질 및 복잡도 분석
        #     total_fields = 0
        #     null_fields = 0
        #     complex_fields = 0
        #     requires_validation = False
        #
        #     for item in items[:10]:  # 최대 10개 샘플만 확인
        #         for key, value in item.items():
        #             total_fields += 1
        #             if value is None:
        #                 null_fields += 1
        #             elif isinstance(value, (dict, list)):
        #                 complex_fields += 1
        #             # 모호한 필드나 정책 결정이 필요한 경우 체크
        #             if key in ["nickname", "e_player_name"] and value:
        #                 requires_validation = True
        #
        #     null_ratio = null_fields / total_fields if total_fields > 0 else 0
        #     complexity_ratio = complex_fields / total_fields if total_fields > 0 else 0
        #
        #     # 정책 기반이 필요한 경우 (예: 복잡한 검증, 데이터 보강 등)
        #     if requires_validation and null_ratio < 0.3:
        #         logger.info("[판단] 정책 기반 선택 (복잡한 검증 필요)")
        #         return "policy"
        #     else:
        #         logger.info("[판단] 규칙 기반 선택 (단순한 데이터 삽입)")
        #         return "rule"
        #
        # except Exception as e:
        #     logger.error(f"[판단] 오류 발생, 규칙 기반 사용: {e}", exc_info=True)
        #     return "rule"

    async def process_players(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """선수 데이터를 처리합니다.

        정책기반/규칙기반을 판단하여 적절한 전략을 선택합니다.

        Args:
            items: 처리할 선수 데이터 리스트

        Returns:
            처리 결과 딕셔너리
        """
        logger.info(f"[오케스트레이터] 라우터로부터 {len(items)}개 항목 수신")

        # 상위 5개 데이터 출력
        logger.info("[오케스트레이터] 수신된 데이터 상위 5개 출력:")
        top_five_items = items[:5]
        for idx, item in enumerate(top_five_items, start=1):
            logger.info(f"  [오케스트레이터 {idx}] {json.dumps(item, ensure_ascii=False, indent=2)}")

        logger.info(f"[오케스트레이터] {len(items)}개 항목 처리 시작")

        # 전략 타입 판단
        strategy_type = self._determine_strategy_type(items)
        logger.info(f"[오케스트레이터] 선택된 전략: {strategy_type}")

        # 전략 선택 및 실행
        if strategy_type == "policy":
            strategy = self.policy_strategy
        else:
            strategy = self.rule_strategy

        result = await strategy.process(items)

        # 결과에 전략 정보 추가
        result["strategy_used"] = strategy_type
        result["total_items"] = len(items)

        logger.info(f"[오케스트레이터] 처리 완료: {strategy_type} 전략 사용")
        return result
