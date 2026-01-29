"""경기장 데이터 정책 기반 에이전트."""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from fastmcp import FastMCP
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    from langchain_community.llms import HuggingFacePipeline

from app.core.llm.providers.exaone_local import create_exaone_local_llm

logger = logging.getLogger(__name__)


class StadiumAgent:
    """경기장 데이터를 정책 기반으로 처리하는 에이전트."""

    def __init__(self, model_dir: Optional[Path] = None):
        """StadiumAgent 초기화.

        Args:
            model_dir: ExaOne 모델 디렉토리 경로 (None이면 기본 경로 사용)
        """
        logger.info("[에이전트] StadiumAgent 초기화")

        # ExaOne 모델 로드
        self.exaone_llm = self._load_exaone_model(model_dir)

        # FastMCP 클라이언트 생성 및 툴 설정
        self.mcp = FastMCP(name="stadium_agent_exaone")
        self._setup_exaone_tools()

        logger.info("[에이전트] StadiumAgent 초기화 완료 (ExaOne, FastMCP)")

    def _get_default_model_dir(self) -> Path:
        """기본 ExaOne 모델 디렉토리 경로를 반환합니다.

        Returns:
            모델 디렉토리 Path
        """
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent.parent.parent
        model_dir = project_root / "artifacts" / "exaone-2.4b"
        return model_dir

    def _load_exaone_model(self, model_dir: Optional[Path] = None):
        """ExaOne 모델을 로드합니다.

        Args:
            model_dir: 모델 디렉토리 경로

        Returns:
            LangChain 호환 LLM 인스턴스
        """
        if model_dir is None:
            model_dir = self._get_default_model_dir()

        if not model_dir.exists():
            logger.warning(f"[ExaOne] 모델 디렉토리를 찾을 수 없습니다: {model_dir}")
            logger.info("[ExaOne] 기본 경로에서 모델 로드 시도")
            try:
                return create_exaone_local_llm()
            except Exception as e:
                logger.error(f"[ExaOne] 모델 로드 실패: {e}", exc_info=True)
                raise

        logger.info(f"[ExaOne] 모델 로딩 중: {model_dir}")

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[ExaOne] 사용 디바이스: {device}")

            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                local_files_only=True
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 모델 로드 설정
            model_kwargs = {
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "device_map": "auto" if device == "cuda" else None,
                "trust_remote_code": True,
                "local_files_only": True
            }

            # 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                **model_kwargs
            )

            # 텍스트 생성 파이프라인 생성
            text_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                device=0 if device == "cuda" else -1,
            )

            # LangChain HuggingFacePipeline로 래핑
            llm = HuggingFacePipeline(
                pipeline=text_pipeline,
                model_kwargs={
                    "temperature": 0.7,
                    "max_new_tokens": 512,
                    "do_sample": True,
                    "top_p": 0.9,
                }
            )

            logger.info("[ExaOne] 모델 로딩 완료")
            return llm

        except Exception as e:
            logger.error(f"[ExaOne] 모델 로딩 실패: {e}", exc_info=True)
            raise RuntimeError(f"ExaOne 모델 로딩 실패: {e}") from e

    def _setup_exaone_tools(self) -> None:
        """ExaOne 모델을 위한 FastMCP 툴을 설정합니다."""
        @self.mcp.tool()
        def exaone_generate_text(prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
            """ExaOne 모델을 사용하여 텍스트를 생성합니다.

            Args:
                prompt: 생성할 텍스트의 프롬프트
                max_tokens: 최대 생성 토큰 수

            Returns:
                생성 결과 딕셔너리
            """
            try:
                formatted_prompt = f"[질문] {prompt}\n[답변] "
                response = self.exaone_llm.invoke(formatted_prompt)

                # 응답에서 프롬프트 부분 제거
                if "[답변]" in response:
                    response = response.split("[답변]")[-1].strip()

                logger.info(f"[ExaOne 툴] 텍스트 생성 완료: {len(response)}자")
                return {
                    "success": True,
                    "generated_text": response,
                    "prompt": prompt,
                    "length": len(response)
                }
            except Exception as e:
                logger.error(f"[ExaOne 툴] 텍스트 생성 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.mcp.tool()
        def exaone_analyze_stadium_data(stadium_data: Dict[str, Any]) -> Dict[str, Any]:
            """ExaOne 모델을 사용하여 경기장 데이터를 분석합니다.

            Args:
                stadium_data: 분석할 경기장 데이터 딕셔너리

            Returns:
                분석 결과 딕셔너리
            """
            try:
                # 경기장 데이터를 텍스트로 변환
                data_text = json.dumps(stadium_data, ensure_ascii=False, indent=2)
                prompt = f"다음 경기장 데이터를 분석하고 주요 특징을 요약해주세요:\n\n{data_text}"

                # ExaOne 모델 직접 호출
                formatted_prompt = f"[질문] {prompt}\n[답변] "
                response = self.exaone_llm.invoke(formatted_prompt)

                if "[답변]" in response:
                    response = response.split("[답변]")[-1].strip()

                logger.info("[ExaOne 툴] 경기장 데이터 분석 완료")
                return {
                    "success": True,
                    "analysis": response,
                    "stadium_data": stadium_data
                }
            except Exception as e:
                logger.error(f"[ExaOne 툴] 경기장 데이터 분석 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        logger.info("[FastMCP] ExaOne 툴 설정 완료")

    async def process_stadiums(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """경기장 데이터를 정책 기반으로 처리합니다.

        Args:
            items: 처리할 경기장 데이터 리스트

        Returns:
            처리 결과 딕셔너리
        """
        logger.info(f"[에이전트] 정책 기반 처리 시작: {len(items)}개 항목")

        # TODO: 정책 기반 처리 로직 구현
        # 예: LLM을 사용한 데이터 검증, 변환, 보강 등

        processed_items = []
        for item in items:
            # 정책 기반 처리 예시
            processed_item = {
                **item,
                "processed_by": "policy_agent",
                "policy_applied": True,
            }
            processed_items.append(processed_item)

        result = {
            "success": True,
            "method": "policy_based",
            "processed_count": len(processed_items),
            "items": processed_items,
        }

        logger.info(f"[에이전트] 정책 기반 처리 완료: {len(processed_items)}개 항목")
        return result

