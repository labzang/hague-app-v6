"""SFT 데이터셋의 품질 검증, 정제, 토크나이징 및 Train/Validation 분할 모듈."""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from transformers import AutoTokenizer
    import torch
    from datasets import Dataset, load_dataset
except ImportError:
    print("오류: transformers 또는 datasets 패키지가 설치되지 않았습니다.")
    print("pip install transformers torch datasets 를 실행하세요.")
    sys.exit(1)


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """JSONL 파일을 읽어서 리스트로 반환합니다.

    Args:
        file_path: JSONL 파일 경로

    Returns:
        파싱된 JSON 객체 리스트
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"경고: {line_num}번째 줄 JSON 파싱 오류: {e}")
                continue
    return data


def format_sft_text(example: Dict[str, Any]) -> str:
    """SFT 예제를 학습용 텍스트 형식으로 변환합니다.

    Args:
        example: SFT 형식의 예제 딕셔너리

    Returns:
        포맷된 텍스트 문자열
    """
    instruction = example.get("instruction", "")
    input_data = example.get("input", {})
    output = example.get("output", {})

    # input을 JSON 문자열로 변환
    input_str = json.dumps(input_data, ensure_ascii=False)
    # output을 JSON 문자열로 변환
    output_str = json.dumps(output, ensure_ascii=False)

    # EXAONE 모델에 맞는 프롬프트 템플릿
    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_str}\n\n### Response:\n{output_str}"
    return text


def validate_data_quality(
    examples: List[Dict[str, Any]], verbose: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """데이터 품질을 검증하고 정제합니다.

    Args:
        examples: 검증할 예제 리스트
        verbose: 상세 정보 출력 여부

    Returns:
        (정제된 예제 리스트, 통계 딕셔너리) 튜플
    """
    stats = {
        "total": len(examples),
        "valid": 0,
        "invalid_missing_fields": 0,
        "invalid_empty_fields": 0,
        "invalid_json_format": 0,
        "removed": 0,
    }

    cleaned_examples = []

    for i, example in enumerate(examples):
        # 1. 필수 필드 검증
        required_fields = ["instruction", "input", "output"]
        if not all(field in example for field in required_fields):
            stats["invalid_missing_fields"] += 1
            if verbose and stats["invalid_missing_fields"] <= 5:
                print(f"경고 [{i+1}]: 필수 필드 누락 - {required_fields}")
            continue

        # 2. 빈 값 검증
        instruction = example.get("instruction", "").strip()
        input_data = example.get("input")
        output = example.get("output")

        if not instruction:
            stats["invalid_empty_fields"] += 1
            if verbose and stats["invalid_empty_fields"] <= 5:
                print(f"경고 [{i+1}]: instruction 필드가 비어있음")
            continue

        if not input_data or not isinstance(input_data, dict):
            stats["invalid_empty_fields"] += 1
            if verbose and stats["invalid_empty_fields"] <= 5:
                print(f"경고 [{i+1}]: input 필드가 비어있거나 딕셔너리가 아님")
            continue

        if not output or not isinstance(output, dict):
            stats["invalid_empty_fields"] += 1
            if verbose and stats["invalid_empty_fields"] <= 5:
                print(f"경고 [{i+1}]: output 필드가 비어있거나 딕셔너리가 아님")
            continue

        # 3. output의 필수 필드 검증 (action, reason, confidence)
        if "action" not in output or "reason" not in output:
            stats["invalid_json_format"] += 1
            if verbose and stats["invalid_json_format"] <= 5:
                print(f"경고 [{i+1}]: output에 필수 필드(action, reason) 누락")
            continue

        # 4. action 값 검증
        if output.get("action") not in ["BLOCK", "ALLOW"]:
            stats["invalid_json_format"] += 1
            if verbose and stats["invalid_json_format"] <= 5:
                print(f"경고 [{i+1}]: action 값이 유효하지 않음: {output.get('action')}")
            continue

        # 5. confidence 값 검증
        confidence = output.get("confidence")
        if confidence is None or not isinstance(confidence, (int, float)):
            stats["invalid_json_format"] += 1
            if verbose and stats["invalid_json_format"] <= 5:
                print(f"경고 [{i+1}]: confidence 값이 유효하지 않음: {confidence}")
            continue

        # 모든 검증 통과
        cleaned_examples.append(example)
        stats["valid"] += 1

    stats["removed"] = stats["total"] - stats["valid"]

    if verbose:
        print("\n=== 데이터 품질 검증 결과 ===")
        print(f"전체 샘플: {stats['total']}")
        print(f"유효 샘플: {stats['valid']}")
        print(f"제거된 샘플: {stats['removed']}")
        print(f"  - 필수 필드 누락: {stats['invalid_missing_fields']}")
        print(f"  - 빈 값: {stats['invalid_empty_fields']}")
        print(f"  - JSON 형식 오류: {stats['invalid_json_format']}")
        print()

    return cleaned_examples, stats


def filter_by_token_length(
    examples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """토큰 길이를 기준으로 샘플을 필터링합니다.

    Args:
        examples: 필터링할 예제 리스트
        tokenizer: 토크나이저 인스턴스
        max_length: 최대 토큰 길이
        verbose: 상세 정보 출력 여부

    Returns:
        (필터링된 예제 리스트, 통계 딕셔너리) 튜플
    """
    stats = {
        "total": len(examples),
        "within_limit": 0,
        "exceeded_limit": 0,
        "max_token_length": 0,
        "min_token_length": float("inf"),
        "avg_token_length": 0.0,
    }

    filtered_examples = []
    token_lengths = []

    for example in examples:
        # 텍스트로 변환
        text = format_sft_text(example)

        # 토크나이징
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_count = len(tokens)

        token_lengths.append(token_count)
        stats["max_token_length"] = max(stats["max_token_length"], token_count)
        stats["min_token_length"] = min(stats["min_token_length"], token_count)

        if token_count <= max_length:
            # 토큰 길이 정보를 메타데이터로 추가
            example["_token_length"] = token_count
            filtered_examples.append(example)
            stats["within_limit"] += 1
        else:
            stats["exceeded_limit"] += 1
            if verbose and stats["exceeded_limit"] <= 5:
                print(
                    f"경고: 토큰 길이 초과 ({token_count} > {max_length}): "
                    f"제목={example.get('input', {}).get('subject', '')[:50]}"
                )

    if token_lengths:
        stats["avg_token_length"] = sum(token_lengths) / len(token_lengths)

    if verbose:
        print("\n=== 토큰 길이 필터링 결과 ===")
        print(f"최대 시퀀스 길이 제한: {max_length} 토큰")
        print(f"전체 샘플: {stats['total']}")
        print(f"제한 내 샘플: {stats['within_limit']}")
        print(f"초과 샘플: {stats['exceeded_limit']}")
        print(f"최소 토큰 길이: {stats['min_token_length']}")
        print(f"최대 토큰 길이: {stats['max_token_length']}")
        print(f"평균 토큰 길이: {stats['avg_token_length']:.2f}")
        print()

    return filtered_examples, stats


def stratified_split(
    examples: List[Dict[str, Any]],
    train_ratio: float = 0.9,
    random_seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified 방식으로 Train/Validation 세트를 분할합니다.

    BLOCK/ALLOW 비율을 유지하면서 분할합니다.

    Args:
        examples: 분할할 예제 리스트
        train_ratio: Train 세트 비율 (0.0 ~ 1.0)
        random_seed: 랜덤 시드
        verbose: 상세 정보 출력 여부

    Returns:
        (train_examples, val_examples) 튜플
    """
    import random

    random.seed(random_seed)

    # action 기준으로 그룹화
    block_examples = []
    allow_examples = []

    for example in examples:
        action = example.get("output", {}).get("action", "")
        if action == "BLOCK":
            block_examples.append(example)
        elif action == "ALLOW":
            allow_examples.append(example)

    # 각 그룹을 무작위로 섞기
    random.shuffle(block_examples)
    random.shuffle(allow_examples)

    # 비율에 따라 분할
    block_train_size = int(len(block_examples) * train_ratio)
    allow_train_size = int(len(allow_examples) * train_ratio)

    train_examples = (
        block_examples[:block_train_size] + allow_examples[:allow_train_size]
    )
    val_examples = (
        block_examples[block_train_size:] + allow_examples[allow_train_size:]
    )

    # Train 세트도 무작위로 섞기
    random.shuffle(train_examples)
    random.shuffle(val_examples)

    if verbose:
        print("\n=== Train/Validation 분할 결과 ===")
        print(f"Train 비율: {train_ratio * 100:.1f}%")
        print(f"Validation 비율: {(1 - train_ratio) * 100:.1f}%")
        print()

        # Train 세트 통계
        train_actions = Counter(
            ex.get("output", {}).get("action", "") for ex in train_examples
        )
        print(f"Train 세트: {len(train_examples)}개")
        for action, count in train_actions.items():
            ratio = count / len(train_examples) * 100
            print(f"  {action}: {count}개 ({ratio:.1f}%)")

        # Validation 세트 통계
        val_actions = Counter(
            ex.get("output", {}).get("action", "") for ex in val_examples
        )
        print(f"\nValidation 세트: {len(val_examples)}개")
        for action, count in val_actions.items():
            ratio = count / len(val_examples) * 100 if val_examples else 0
            print(f"  {action}: {count}개 ({ratio:.1f}%)")
        print()

    return train_examples, val_examples


def save_jsonl(examples: List[Dict[str, Any]], file_path: Path) -> None:
    """예제 리스트를 JSONL 파일로 저장합니다.

    Args:
        examples: 저장할 예제 리스트
        file_path: 저장할 파일 경로
    """
    # 메타데이터 필드 제거 (토큰 길이 등)
    cleaned_examples = []
    for example in examples:
        cleaned = {k: v for k, v in example.items() if not k.startswith("_")}
        cleaned_examples.append(cleaned)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for example in cleaned_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def create_datasets_from_examples(
    train_examples: List[Dict[str, Any]],
    val_examples: List[Dict[str, Any]],
    verbose: bool = True,
) -> Tuple[Dataset, Dataset]:
    """예제 리스트를 HuggingFace Dataset 객체로 변환합니다.

    Trainer/TRL의 SFTTrainer가 요구하는 Dataset 형식으로 변환합니다.
    shuffle, batch, collate, multiprocessing, cache가 자동으로 처리됩니다.

    Args:
        train_examples: Train 예제 리스트
        val_examples: Validation 예제 리스트
        verbose: 상세 정보 출력 여부

    Returns:
        (train_dataset, val_dataset) 튜플
    """
    if verbose:
        print("\n[Dataset 변환] HuggingFace Dataset 객체 생성 중...")

    # 메타데이터 필드 제거 후 Dataset 생성
    def clean_example(example: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in example.items() if not k.startswith("_")}

    train_cleaned = [clean_example(ex) for ex in train_examples]
    val_cleaned = [clean_example(ex) for ex in val_examples]

    # Dataset.from_list()를 사용하여 Dataset 객체 생성
    train_dataset = Dataset.from_list(train_cleaned)
    val_dataset = Dataset.from_list(val_cleaned)

    # 텍스트 포맷으로 변환하는 함수
    def format_text(example: Dict[str, Any]) -> Dict[str, str]:
        """SFT 예제를 학습용 텍스트 형식으로 변환합니다."""
        text = format_sft_text(example)
        return {"text": text}

    # Dataset에 text 필드 추가 (SFTTrainer가 요구하는 형식)
    train_dataset = train_dataset.map(format_text, desc="Train 텍스트 포맷 변환")
    val_dataset = val_dataset.map(format_text, desc="Validation 텍스트 포맷 변환")

    if verbose:
        print(f"Train Dataset: {len(train_dataset)}개 샘플")
        print(f"  - 컬럼: {train_dataset.column_names}")
        print(f"  - 특징: {train_dataset.features}")
        print(f"Validation Dataset: {len(val_dataset)}개 샘플")
        print(f"  - 컬럼: {val_dataset.column_names}")
        print(f"  - 특징: {val_dataset.features}")
        print()

    return train_dataset, val_dataset


def save_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """Dataset 객체를 디스크에 저장합니다.

    Args:
        train_dataset: Train Dataset 객체
        val_dataset: Validation Dataset 객체
        output_dir: 저장할 디렉토리 경로
        verbose: 상세 정보 출력 여부
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset_path = output_dir / "train_dataset"
    val_dataset_path = output_dir / "val_dataset"

    if verbose:
        print("\n[Dataset 저장] Dataset 객체 저장 중...")

    train_dataset.save_to_disk(str(train_dataset_path))
    val_dataset.save_to_disk(str(val_dataset_path))

    if verbose:
        print(f"Train Dataset 저장: {train_dataset_path}")
        print(f"Validation Dataset 저장: {val_dataset_path}")


def load_datasets(dataset_dir: Path) -> Tuple[Dataset, Dataset]:
    """저장된 Dataset 객체를 로드합니다.

    Args:
        dataset_dir: Dataset이 저장된 디렉토리 경로

    Returns:
        (train_dataset, val_dataset) 튜플
    """
    train_dataset_path = dataset_dir / "train_dataset"
    val_dataset_path = dataset_dir / "val_dataset"

    train_dataset = Dataset.load_from_disk(str(train_dataset_path))
    val_dataset = Dataset.load_from_disk(str(val_dataset_path))

    return train_dataset, val_dataset


def process_sft_dataset(
    input_jsonl_path: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    train_ratio: float = 0.9,
    max_seq_length: int = 2048,
    random_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """SFT 데이터셋을 처리하는 전체 파이프라인.

    Args:
        input_jsonl_path: 입력 JSONL 파일 경로
        model_dir: 모델 디렉토리 경로 (토크나이저 로드용)
        output_dir: 출력 디렉토리 경로
        train_ratio: Train 세트 비율
        max_seq_length: 최대 시퀀스 길이 (토큰)
        random_seed: 랜덤 시드
        verbose: 상세 정보 출력 여부

    Returns:
        처리 통계 딕셔너리
    """
    if verbose:
        print("=" * 60)
        print("SFT 데이터셋 처리 파이프라인 시작")
        print("=" * 60)

    # 1. 데이터 로드
    if verbose:
        print(f"\n[1/5] 데이터 로드 중: {input_jsonl_path}")
    examples = load_jsonl(input_jsonl_path)
    print(f"로드된 샘플 수: {len(examples)}")

    # 2. 데이터 품질 검증 및 정제
    if verbose:
        print("\n[2/7] 데이터 품질 검증 및 정제 중...")
    cleaned_examples, quality_stats = validate_data_quality(examples, verbose=verbose)

    # 3. 토크나이저 로드
    if verbose:
        print(f"\n[3/7] 토크나이저 로드 중: {model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("토크나이저 로드 완료")
    except Exception as e:
        print(f"오류: 토크나이저 로드 실패: {e}")
        raise

    # 4. 토큰 길이 필터링
    if verbose:
        print(f"\n[4/7] 토큰 길이 필터링 중 (최대 {max_seq_length} 토큰)...")
    filtered_examples, token_stats = filter_by_token_length(
        cleaned_examples, tokenizer, max_length=max_seq_length, verbose=verbose
    )

    # 5. Train/Validation 분할
    if verbose:
        print("\n[5/7] Train/Validation 분할 중...")
    train_examples, val_examples = stratified_split(
        filtered_examples, train_ratio=train_ratio, random_seed=random_seed, verbose=verbose
    )

    # 6. JSONL 파일 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    if verbose:
        print(f"\n[JSONL 저장] JSONL 파일 저장 중...")
    save_jsonl(train_examples, train_path)
    save_jsonl(val_examples, val_path)

    if verbose:
        print(f"Train 세트 저장: {train_path} ({len(train_examples)}개)")
        print(f"Validation 세트 저장: {val_path} ({len(val_examples)}개)")

    # 7. Dataset 객체 생성 및 저장
    train_dataset, val_dataset = create_datasets_from_examples(
        train_examples, val_examples, verbose=verbose
    )
    save_datasets(train_dataset, val_dataset, output_dir, verbose=verbose)

    # 통계 요약
    summary = {
        "input_file": str(input_jsonl_path),
        "output_dir": str(output_dir),
        "quality_stats": quality_stats,
        "token_stats": token_stats,
        "train_samples": len(train_examples),
        "val_samples": len(val_examples),
        "train_ratio": train_ratio,
        "max_seq_length": max_seq_length,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("처리 완료!")
        print("=" * 60)
        print(f"최종 Train 샘플: {len(train_examples)}개")
        print(f"최종 Validation 샘플: {len(val_examples)}개")
        print()

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SFT 데이터셋의 품질 검증, 정제, 토크나이징 및 Train/Validation 분할"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="app/data/한국우편사업진흥원_스팸메일 수신차단 목록_20241231.sft.jsonl",
        help="입력 JSONL 파일 경로",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="app/models/exaone-2.4b",
        help="모델 디렉토리 경로 (토크나이저 로드용)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="app/data/spam_agent_processed",
        help="출력 디렉토리 경로",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train 세트 비율 (기본값: 0.9)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="최대 시퀀스 길이 (토큰, 기본값: 2048)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본값: 42)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"오류: 입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    if not model_dir.exists():
        print(f"오류: 모델 디렉토리를 찾을 수 없습니다: {model_dir}")
        sys.exit(1)

    try:
        summary = process_sft_dataset(
            input_jsonl_path=input_path,
            model_dir=model_dir,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            max_seq_length=args.max_seq_length,
            random_seed=args.random_seed,
            verbose=True,
        )
        print("\n처리 성공!")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

