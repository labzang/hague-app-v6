"""축구 도메인 중앙 MCP 서버.

축구 도메인 전용 LLM 모델(ExaOne, KoELECTRA)과 툴을 중앙에서 관리합니다.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from fastmcp import FastMCP
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline

try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    from langchain_community.llms import HuggingFacePipeline

from app.core.llm.providers.exaone_local import create_exaone_local_llm

logger = logging.getLogger(__name__)


class SoccerCentralMCPServer:
    """축구 도메인 중앙 MCP 서버.

    축구 도메인 전용 LLM 모델과 툴을 중앙에서 관리합니다.
    """

    _instance: Optional["SoccerCentralMCPServer"] = None
    _initialized: bool = False

    def __new__(cls):
        """싱글톤 패턴."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """SoccerCentralMCPServer 초기화."""
        if self._initialized:
            return

        logger.info("[축구 중앙 MCP 서버] 초기화 시작")

        # FastMCP 서버 생성
        self.mcp = FastMCP(name="soccer_central_mcp_server")

        # 모델 경로 설정
        self._setup_paths()

        # 모델 로드 (지연 로딩)
        self.exaone_llm: Optional[Any] = None
        self.koelectra_model: Optional[AutoModel] = None
        self.koelectra_tokenizer: Optional[AutoTokenizer] = None

        # 툴 저장소 (직접 호출용)
        self._tools: Dict[str, Any] = {}

        # 툴 설정
        self._setup_exaone_tools()
        self._setup_koelectra_tools()
        self._setup_filesystem_tools()
        self._setup_integrated_tools()

        self._initialized = True
        logger.info("[축구 중앙 MCP 서버] 초기화 완료")

    def _setup_paths(self) -> None:
        """경로 설정."""
        current_file = Path(__file__)
        # app/domain/v10/soccer/hub/mcp/central_mcp_server.py
        # -> mcp -> hub -> soccer -> v10 -> domain -> app -> 프로젝트 루트 (7단계 위)
        project_root = current_file.parent.parent.parent.parent.parent.parent.parent
        self.project_root = project_root
        self.exaone_model_dir = project_root / "artifacts" / "base-models" / "exaone-2.4b"
        self.koelectra_model_dir = project_root / "artifacts" / "models--monologg--koelectra-small-v3-discriminator"

    def _load_exaone_model(self):
        """ExaOne 모델을 로드합니다 (지연 로딩)."""
        if self.exaone_llm is None:
            logger.info("[축구 중앙 MCP 서버] ExaOne 모델 로딩 중...")
            if not self.exaone_model_dir.exists():
                logger.warning(f"[축구 중앙 MCP 서버] ExaOne 모델 디렉토리를 찾을 수 없습니다: {self.exaone_model_dir}")
                try:
                    self.exaone_llm = create_exaone_local_llm()
                except Exception as e:
                    logger.error(f"[축구 중앙 MCP 서버] ExaOne 모델 로드 실패: {e}", exc_info=True)
                    raise
            else:
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"[축구 중앙 MCP 서버] ExaOne 사용 디바이스: {device}")

                    tokenizer = AutoTokenizer.from_pretrained(
                        str(self.exaone_model_dir),
                        trust_remote_code=True,
                        local_files_only=True
                    )

                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token

                    model_kwargs = {
                        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                        "device_map": "auto" if device == "cuda" else None,
                        "trust_remote_code": True,
                        "local_files_only": True
                    }

                    model = AutoModelForCausalLM.from_pretrained(
                        str(self.exaone_model_dir),
                        **model_kwargs
                    )

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

                    self.exaone_llm = HuggingFacePipeline(
                        pipeline=text_pipeline,
                        model_kwargs={
                            "temperature": 0.7,
                            "max_new_tokens": 512,
                            "do_sample": True,
                            "top_p": 0.9,
                        }
                    )

                    logger.info("[축구 중앙 MCP 서버] ExaOne 모델 로딩 완료")
                except Exception as e:
                    logger.error(f"[축구 중앙 MCP 서버] ExaOne 모델 로딩 실패: {e}", exc_info=True)
                    raise
        return self.exaone_llm

    def _load_koelectra_model(self) -> tuple[AutoModel, AutoTokenizer]:
        """KoELECTRA 모델을 로드합니다 (지연 로딩)."""
        if self.koelectra_model is None or self.koelectra_tokenizer is None:
            logger.info("[축구 중앙 MCP 서버] KoELECTRA 모델 로딩 중...")
            if not self.koelectra_model_dir.exists():
                raise FileNotFoundError(f"KoELECTRA 모델 디렉토리를 찾을 수 없습니다: {self.koelectra_model_dir}")

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(self.koelectra_model_dir),
                    local_files_only=True,
                )
                logger.info("[축구 중앙 MCP 서버] KoELECTRA 토크나이저 로드 완료")

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = AutoModel.from_pretrained(
                    str(self.koelectra_model_dir),
                    local_files_only=True,
                ).to(device)
                model.eval()
                logger.info(f"[축구 중앙 MCP 서버] KoELECTRA 모델 로드 완료 (디바이스: {device})")

                self.koelectra_model = model
                self.koelectra_tokenizer = tokenizer
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] KoELECTRA 모델 로딩 실패: {e}", exc_info=True)
                raise RuntimeError(f"KoELECTRA 모델 로딩 실패: {e}") from e

        return self.koelectra_model, self.koelectra_tokenizer

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
                llm = self._load_exaone_model()
                formatted_prompt = f"[질문] {prompt}\n[답변] "
                response = llm.invoke(formatted_prompt)

                if "[답변]" in response:
                    response = response.split("[답변]")[-1].strip()

                logger.info(f"[축구 중앙 MCP 서버] ExaOne 텍스트 생성 완료: {len(response)}자")
                return {
                    "success": True,
                    "generated_text": response,
                    "prompt": prompt,
                    "length": len(response)
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] ExaOne 텍스트 생성 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.mcp.tool()
        def exaone_analyze_player_data(player_data: Dict[str, Any]) -> Dict[str, Any]:
            """ExaOne 모델을 사용하여 선수 데이터를 분석합니다."""
            try:
                data_text = json.dumps(player_data, ensure_ascii=False, indent=2)
                prompt = f"다음 선수 데이터를 분석하고 주요 특징을 요약해주세요:\n\n{data_text}"

                llm = self._load_exaone_model()
                formatted_prompt = f"[질문] {prompt}\n[답변] "
                response = llm.invoke(formatted_prompt)

                if "[답변]" in response:
                    response = response.split("[답변]")[-1].strip()

                logger.info("[축구 중앙 MCP 서버] ExaOne 선수 데이터 분석 완료")
                return {
                    "success": True,
                    "analysis": response,
                    "player_data": player_data
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] ExaOne 선수 데이터 분석 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.mcp.tool()
        def exaone_analyze_team_data(team_data: Dict[str, Any]) -> Dict[str, Any]:
            """ExaOne 모델을 사용하여 팀 데이터를 분석합니다."""
            try:
                data_text = json.dumps(team_data, ensure_ascii=False, indent=2)
                prompt = f"다음 팀 데이터를 분석하고 주요 특징을 요약해주세요:\n\n{data_text}"

                llm = self._load_exaone_model()
                formatted_prompt = f"[질문] {prompt}\n[답변] "
                response = llm.invoke(formatted_prompt)

                if "[답변]" in response:
                    response = response.split("[답변]")[-1].strip()

                logger.info("[축구 중앙 MCP 서버] ExaOne 팀 데이터 분석 완료")
                return {
                    "success": True,
                    "analysis": response,
                    "team_data": team_data
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] ExaOne 팀 데이터 분석 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.mcp.tool()
        def exaone_analyze_schedule_data(schedule_data: Dict[str, Any]) -> Dict[str, Any]:
            """ExaOne 모델을 사용하여 경기 일정 데이터를 분석합니다."""
            try:
                data_text = json.dumps(schedule_data, ensure_ascii=False, indent=2)
                prompt = f"다음 경기 일정 데이터를 분석하고 주요 특징을 요약해주세요:\n\n{data_text}"

                llm = self._load_exaone_model()
                formatted_prompt = f"[질문] {prompt}\n[답변] "
                response = llm.invoke(formatted_prompt)

                if "[답변]" in response:
                    response = response.split("[답변]")[-1].strip()

                logger.info("[축구 중앙 MCP 서버] ExaOne 경기 일정 데이터 분석 완료")
                return {
                    "success": True,
                    "analysis": response,
                    "schedule_data": schedule_data
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] ExaOne 경기 일정 데이터 분석 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.mcp.tool()
        def exaone_analyze_stadium_data(stadium_data: Dict[str, Any]) -> Dict[str, Any]:
            """ExaOne 모델을 사용하여 경기장 데이터를 분석합니다."""
            try:
                data_text = json.dumps(stadium_data, ensure_ascii=False, indent=2)
                prompt = f"다음 경기장 데이터를 분석하고 주요 특징을 요약해주세요:\n\n{data_text}"

                llm = self._load_exaone_model()
                formatted_prompt = f"[질문] {prompt}\n[답변] "
                response = llm.invoke(formatted_prompt)

                if "[답변]" in response:
                    response = response.split("[답변]")[-1].strip()

                logger.info("[축구 중앙 MCP 서버] ExaOne 경기장 데이터 분석 완료")
                return {
                    "success": True,
                    "analysis": response,
                    "stadium_data": stadium_data
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] ExaOne 경기장 데이터 분석 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        # 툴 등록
        self._tools["exaone_generate_text"] = exaone_generate_text
        self._tools["exaone_analyze_player_data"] = exaone_analyze_player_data
        self._tools["exaone_analyze_team_data"] = exaone_analyze_team_data
        self._tools["exaone_analyze_schedule_data"] = exaone_analyze_schedule_data
        self._tools["exaone_analyze_stadium_data"] = exaone_analyze_stadium_data

        logger.info("[축구 중앙 MCP 서버] ExaOne 툴 설정 완료")

    def _setup_koelectra_tools(self) -> None:
        """KoELECTRA 모델을 위한 FastMCP 툴을 설정합니다."""
        @self.mcp.tool()
        def koelectra_embed_text(text: str) -> Dict[str, Any]:
            """KoELECTRA 모델을 사용하여 텍스트를 임베딩으로 변환합니다."""
            try:
                model, tokenizer = self._load_koelectra_model()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

                logger.info(f"[축구 중앙 MCP 서버] KoELECTRA 텍스트 임베딩 생성 완료: {len(embedding)}차원")
                return {
                    "success": True,
                    "embedding": embedding,
                    "dimension": len(embedding),
                    "text_length": len(text)
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] KoELECTRA 임베딩 생성 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.mcp.tool()
        def koelectra_classify_text(text: str) -> Dict[str, Any]:
            """KoELECTRA 모델을 사용하여 텍스트를 분류합니다."""
            try:
                model, tokenizer = self._load_koelectra_model()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

                logger.info(f"[축구 중앙 MCP 서버] KoELECTRA 텍스트 분류 완료")
                return {
                    "success": True,
                    "cls_embedding": cls_embedding,
                    "text": text
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] KoELECTRA 분류 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        # 툴 등록
        self._tools["koelectra_embed_text"] = koelectra_embed_text
        self._tools["koelectra_classify_text"] = koelectra_classify_text

        logger.info("[축구 중앙 MCP 서버] KoELECTRA 툴 설정 완료")

    def _setup_filesystem_tools(self) -> None:
        """os와 pathlib 라이브러리를 사용한 파일 시스템 툴을 설정합니다."""
        project_root = self.project_root

        @self.mcp.tool()
        def path_exists(path: str) -> Dict[str, Any]:
            """경로가 존재하는지 확인합니다."""
            try:
                path_obj = Path(path)
                if not path_obj.is_absolute():
                    path_obj = project_root / path_obj

                try:
                    path_obj.resolve().relative_to(project_root.resolve())
                except ValueError:
                    return {
                        "success": False,
                        "error": "프로젝트 루트 밖의 경로는 접근할 수 없습니다"
                    }

                exists = path_obj.exists()
                is_file = path_obj.is_file() if exists else False
                is_dir = path_obj.is_dir() if exists else False

                return {
                    "success": True,
                    "path": str(path_obj),
                    "exists": exists,
                    "is_file": is_file,
                    "is_dir": is_dir
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] 경로 확인 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.mcp.tool()
        def list_directory(path: str = ".") -> Dict[str, Any]:
            """디렉토리 내용을 나열합니다."""
            try:
                path_obj = Path(path)
                if not path_obj.is_absolute():
                    path_obj = project_root / path_obj

                try:
                    path_obj.resolve().relative_to(project_root.resolve())
                except ValueError:
                    return {
                        "success": False,
                        "error": "프로젝트 루트 밖의 경로는 접근할 수 없습니다"
                    }

                if not path_obj.exists():
                    return {
                        "success": False,
                        "error": "경로가 존재하지 않습니다"
                    }

                if not path_obj.is_dir():
                    return {
                        "success": False,
                        "error": "디렉토리가 아닙니다"
                    }

                items = []
                for item in path_obj.iterdir():
                    items.append({
                        "name": item.name,
                        "is_file": item.is_file(),
                        "is_dir": item.is_dir(),
                        "size": item.stat().st_size if item.is_file() else None
                    })

                return {
                    "success": True,
                    "path": str(path_obj),
                    "items": sorted(items, key=lambda x: (not x["is_dir"], x["name"])),
                    "count": len(items)
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] 디렉토리 나열 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.mcp.tool()
        def read_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
            """파일 내용을 읽습니다."""
            try:
                path_obj = Path(file_path)
                if not path_obj.is_absolute():
                    path_obj = project_root / path_obj

                try:
                    path_obj.resolve().relative_to(project_root.resolve())
                except ValueError:
                    return {
                        "success": False,
                        "error": "프로젝트 루트 밖의 경로는 접근할 수 없습니다"
                    }

                if not path_obj.exists():
                    return {
                        "success": False,
                        "error": "파일이 존재하지 않습니다"
                    }

                if not path_obj.is_file():
                    return {
                        "success": False,
                        "error": "파일이 아닙니다"
                    }

                file_size = path_obj.stat().st_size
                if file_size > 10 * 1024 * 1024:
                    return {
                        "success": False,
                        "error": "파일이 너무 큽니다 (10MB 제한)"
                    }

                content = path_obj.read_text(encoding=encoding)

                return {
                    "success": True,
                    "path": str(path_obj),
                    "content": content,
                    "size": file_size,
                    "encoding": encoding
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] 파일 읽기 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.mcp.tool()
        def get_path_info(path: str) -> Dict[str, Any]:
            """경로의 상세 정보를 조회합니다."""
            try:
                path_obj = Path(path)
                if not path_obj.is_absolute():
                    path_obj = project_root / path_obj

                try:
                    path_obj.resolve().relative_to(project_root.resolve())
                except ValueError:
                    return {
                        "success": False,
                        "error": "프로젝트 루트 밖의 경로는 접근할 수 없습니다"
                    }

                if not path_obj.exists():
                    return {
                        "success": True,
                        "path": str(path_obj),
                        "exists": False,
                        "absolute_path": str(path_obj.resolve())
                    }

                stat_info = path_obj.stat()

                return {
                    "success": True,
                    "path": str(path_obj),
                    "absolute_path": str(path_obj.resolve()),
                    "exists": True,
                    "is_file": path_obj.is_file(),
                    "is_dir": path_obj.is_dir(),
                    "size": stat_info.st_size if path_obj.is_file() else None,
                    "created": stat_info.st_ctime,
                    "modified": stat_info.st_mtime,
                    "parent": str(path_obj.parent),
                    "name": path_obj.name,
                    "stem": path_obj.stem,
                    "suffix": path_obj.suffix
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] 경로 정보 조회 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.mcp.tool()
        def join_paths(*paths: str) -> Dict[str, Any]:
            """여러 경로를 결합합니다."""
            try:
                combined = Path(*paths)
                return {
                    "success": True,
                    "combined_path": str(combined),
                    "parts": list(combined.parts)
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] 경로 결합 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.mcp.tool()
        def get_environment_variable(name: str, default: Optional[str] = None) -> Dict[str, Any]:
            """환경 변수를 읽습니다."""
            try:
                value = os.getenv(name, default)
                return {
                    "success": True,
                    "name": name,
                    "value": value,
                    "exists": name in os.environ
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] 환경 변수 읽기 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.mcp.tool()
        def get_current_directory() -> Dict[str, Any]:
            """현재 작업 디렉토리를 반환합니다."""
            try:
                cwd = Path.cwd()
                return {
                    "success": True,
                    "current_directory": str(cwd),
                    "absolute_path": str(cwd.resolve())
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] 현재 디렉토리 조회 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        # 툴 등록
        self._tools["path_exists"] = path_exists
        self._tools["list_directory"] = list_directory
        self._tools["read_file"] = read_file
        self._tools["get_path_info"] = get_path_info
        self._tools["join_paths"] = join_paths
        self._tools["get_environment_variable"] = get_environment_variable
        self._tools["get_current_directory"] = get_current_directory

        logger.info("[축구 중앙 MCP 서버] 파일시스템 툴 설정 완료")

    def _setup_integrated_tools(self) -> None:
        """KoELECTRA와 ExaOne을 연결하는 통합 FastMCP 툴을 설정합니다."""
        @self.mcp.tool()
        async def koelectra_to_exaone_pipeline(text: str) -> Dict[str, Any]:
            """KoELECTRA로 텍스트를 임베딩한 후 ExaOne으로 분석하는 파이프라인."""
            try:
                logger.info(f"[축구 중앙 MCP 서버] 통합 파이프라인 시작: {text[:50]}...")

                # 1단계: KoELECTRA로 임베딩 생성
                model, tokenizer = self._load_koelectra_model()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

                logger.info(f"[축구 중앙 MCP 서버] KoELECTRA 임베딩 생성 완료: {len(embedding)}차원")

                # 2단계: ExaOne으로 텍스트 분석
                analysis_prompt = f"다음 텍스트를 분석하고 주요 내용을 요약해주세요:\n\n{text}"
                llm = self._load_exaone_model()
                formatted_prompt = f"[질문] {analysis_prompt}\n[답변] "
                exaone_result = llm.invoke(formatted_prompt)

                if "[답변]" in exaone_result:
                    exaone_result = exaone_result.split("[답변]")[-1].strip()

                logger.info("[축구 중앙 MCP 서버] ExaOne 분석 완료")

                return {
                    "success": True,
                    "koelectra_embedding": {
                        "dimension": len(embedding),
                        "sample": embedding[:10]
                    },
                    "exaone_analysis": exaone_result,
                    "original_text": text
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] 통합 파이프라인 처리 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.mcp.tool()
        async def analyze_player_with_models(player_data: Dict[str, Any]) -> Dict[str, Any]:
            """KoELECTRA와 ExaOne을 사용하여 선수 데이터를 종합 분석합니다."""
            try:
                logger.info(f"[축구 중앙 MCP 서버] 선수 데이터 분석 시작: {player_data.get('player_name', 'Unknown')}")

                data_text = json.dumps(player_data, ensure_ascii=False, indent=2)

                # 1단계: KoELECTRA로 데이터 임베딩
                model, tokenizer = self._load_koelectra_model()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = tokenizer(
                    data_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

                # 2단계: ExaOne으로 데이터 분석
                analysis_prompt = (
                    f"다음 선수 데이터를 분석하고 주요 특징, 강점, 약점을 요약해주세요:\n\n{data_text}"
                )
                llm = self._load_exaone_model()
                exaone_result = llm.invoke(f"[질문] {analysis_prompt}\n[답변] ")

                if "[답변]" in exaone_result:
                    exaone_result = exaone_result.split("[답변]")[-1].strip()

                logger.info("[축구 중앙 MCP 서버] 선수 데이터 분석 완료")

                return {
                    "success": True,
                    "player_data": player_data,
                    "koelectra_embedding": {
                        "dimension": len(embedding),
                        "sample": embedding[:10]
                    },
                    "exaone_analysis": exaone_result,
                    "summary": {
                        "embedding_dim": len(embedding),
                        "analysis_length": len(exaone_result)
                    }
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] 선수 데이터 분석 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e),
                    "player_data": player_data
                }

        @self.mcp.tool()
        async def analyze_team_with_models(team_data: Dict[str, Any]) -> Dict[str, Any]:
            """KoELECTRA와 ExaOne을 사용하여 팀 데이터를 종합 분석합니다."""
            try:
                logger.info(f"[축구 중앙 MCP 서버] 팀 데이터 분석 시작: {team_data.get('team_name', 'Unknown')}")

                data_text = json.dumps(team_data, ensure_ascii=False, indent=2)

                # 1단계: KoELECTRA로 데이터 임베딩
                model, tokenizer = self._load_koelectra_model()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = tokenizer(
                    data_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

                # 2단계: ExaOne으로 데이터 분석
                analysis_prompt = (
                    f"다음 팀 데이터를 분석하고 주요 특징, 선수 구성, 전술 정보를 요약해주세요:\n\n{data_text}"
                )
                llm = self._load_exaone_model()
                exaone_result = llm.invoke(f"[질문] {analysis_prompt}\n[답변] ")

                if "[답변]" in exaone_result:
                    exaone_result = exaone_result.split("[답변]")[-1].strip()

                logger.info("[축구 중앙 MCP 서버] 팀 데이터 분석 완료")

                return {
                    "success": True,
                    "team_data": team_data,
                    "koelectra_embedding": {
                        "dimension": len(embedding),
                        "sample": embedding[:10]
                    },
                    "exaone_analysis": exaone_result,
                    "summary": {
                        "embedding_dim": len(embedding),
                        "analysis_length": len(exaone_result)
                    }
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] 팀 데이터 분석 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e),
                    "team_data": team_data
                }

        @self.mcp.tool()
        async def analyze_schedule_with_models(schedule_data: Dict[str, Any]) -> Dict[str, Any]:
            """KoELECTRA와 ExaOne을 사용하여 경기 일정 데이터를 종합 분석합니다."""
            try:
                logger.info(f"[축구 중앙 MCP 서버] 경기 일정 데이터 분석 시작: {schedule_data.get('match_date', 'Unknown')}")

                data_text = json.dumps(schedule_data, ensure_ascii=False, indent=2)

                # 1단계: KoELECTRA로 데이터 임베딩
                model, tokenizer = self._load_koelectra_model()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = tokenizer(
                    data_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

                # 2단계: ExaOne으로 데이터 분석
                analysis_prompt = (
                    f"다음 경기 일정 데이터를 분석하고 주요 특징, 경기 정보를 요약해주세요:\n\n{data_text}"
                )
                llm = self._load_exaone_model()
                exaone_result = llm.invoke(f"[질문] {analysis_prompt}\n[답변] ")

                if "[답변]" in exaone_result:
                    exaone_result = exaone_result.split("[답변]")[-1].strip()

                logger.info("[축구 중앙 MCP 서버] 경기 일정 데이터 분석 완료")

                return {
                    "success": True,
                    "schedule_data": schedule_data,
                    "koelectra_embedding": {
                        "dimension": len(embedding),
                        "sample": embedding[:10]
                    },
                    "exaone_analysis": exaone_result,
                    "summary": {
                        "embedding_dim": len(embedding),
                        "analysis_length": len(exaone_result)
                    }
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] 경기 일정 데이터 분석 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e),
                    "schedule_data": schedule_data
                }

        @self.mcp.tool()
        async def analyze_stadium_with_models(stadium_data: Dict[str, Any]) -> Dict[str, Any]:
            """KoELECTRA와 ExaOne을 사용하여 경기장 데이터를 종합 분석합니다."""
            try:
                logger.info(f"[축구 중앙 MCP 서버] 경기장 데이터 분석 시작: {stadium_data.get('stadium_name', 'Unknown')}")

                data_text = json.dumps(stadium_data, ensure_ascii=False, indent=2)

                # 1단계: KoELECTRA로 데이터 임베딩
                model, tokenizer = self._load_koelectra_model()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = tokenizer(
                    data_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

                # 2단계: ExaOne으로 데이터 분석
                analysis_prompt = (
                    f"다음 경기장 데이터를 분석하고 주요 특징, 수용 인원, 위치 정보를 요약해주세요:\n\n{data_text}"
                )
                llm = self._load_exaone_model()
                exaone_result = llm.invoke(f"[질문] {analysis_prompt}\n[답변] ")

                if "[답변]" in exaone_result:
                    exaone_result = exaone_result.split("[답변]")[-1].strip()

                logger.info("[축구 중앙 MCP 서버] 경기장 데이터 분석 완료")

                return {
                    "success": True,
                    "stadium_data": stadium_data,
                    "koelectra_embedding": {
                        "dimension": len(embedding),
                        "sample": embedding[:10]
                    },
                    "exaone_analysis": exaone_result,
                    "summary": {
                        "embedding_dim": len(embedding),
                        "analysis_length": len(exaone_result)
                    }
                }
            except Exception as e:
                logger.error(f"[축구 중앙 MCP 서버] 경기장 데이터 분석 실패: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e),
                    "stadium_data": stadium_data
                }

        # 툴 등록
        self._tools["koelectra_to_exaone_pipeline"] = koelectra_to_exaone_pipeline
        self._tools["analyze_player_with_models"] = analyze_player_with_models
        self._tools["analyze_team_with_models"] = analyze_team_with_models
        self._tools["analyze_schedule_with_models"] = analyze_schedule_with_models
        self._tools["analyze_stadium_with_models"] = analyze_stadium_with_models

        logger.info("[축구 중앙 MCP 서버] 통합 툴 설정 완료 (KoELECTRA + ExaOne)")

    def get_mcp_server(self) -> FastMCP:
        """MCP 서버 인스턴스를 반환합니다."""
        return self.mcp

    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """툴을 호출합니다 (클라이언트용)."""
        if tool_name not in self._tools:
            return {
                "success": False,
                "error": f"툴을 찾을 수 없습니다: {tool_name}"
            }

        try:
            tool_func = self._tools[tool_name]
            # async 함수인지 확인
            import inspect
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(**kwargs)
            else:
                result = tool_func(**kwargs)
            return result
        except Exception as e:
            logger.error(f"[축구 중앙 MCP 서버] 툴 호출 실패: {tool_name}, {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }


# 전역 싱글톤 인스턴스
_soccer_central_mcp_server: Optional[SoccerCentralMCPServer] = None


def get_soccer_central_mcp_server() -> SoccerCentralMCPServer:
    """축구 도메인 중앙 MCP 서버 싱글톤 인스턴스를 반환합니다."""
    global _soccer_central_mcp_server
    if _soccer_central_mcp_server is None:
        _soccer_central_mcp_server = SoccerCentralMCPServer()
    return _soccer_central_mcp_server

