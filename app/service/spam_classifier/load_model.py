"""KoELECTRA 모델을 로드하는 모듈."""
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
    from transformers import (
        AutoModel,
        AutoTokenizer,
    )
except ImportError as e:
    print(f"오류: 필요한 패키지가 설치되지 않았습니다: {e}")
    print("pip install transformers torch 를 실행하세요.")
    sys.exit(1)


def load_koelectra_model(
    model_dir: Optional[Path] = None,
) -> Tuple[AutoModel, AutoTokenizer]:
    """KoELECTRA 모델과 토크나이저를 로드합니다.

    Args:
        model_dir: 모델 디렉토리 경로 (None이면 기본 경로 사용)

    Returns:
        (model, tokenizer) 튜플

    Raises:
        FileNotFoundError: 모델 디렉토리를 찾을 수 없을 때
        RuntimeError: 모델 로딩 실패 시
    """
    # 기본 모델 경로 설정
    if model_dir is None:
        # app/service/spam_classifier/load_model.py -> app/model/models--monologg--koelectra-small-v3-discriminator
        model_dir = Path(__file__).parent.parent.parent / "model" / "models--monologg--koelectra-small-v3-discriminator"
    else:
        model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {model_dir}")

    print("KoELECTRA 모델 로딩 중...")
    print(f"모델 경로: {model_dir}")

    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        print("토크나이저 로드 완료")

        # 모델 로드
        model = AutoModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        print("모델 로드 완료")

        return model, tokenizer

    except Exception as e:
        print(f"오류 발생: {e}")
        raise RuntimeError(f"모델 로딩 실패: {e}") from e


if __name__ == "__main__":
    # 모델 로드 테스트
    try:
        model, tokenizer = load_koelectra_model()

        print("\n모델과 토크나이저가 성공적으로 로드되었습니다!")
        print(f"모델 타입: {type(model).__name__}")
        print(f"토크나이저 타입: {type(tokenizer).__name__}")

    except Exception as e:
        print(f"\n모델 로딩 실패: {e}")
        sys.exit(1)

