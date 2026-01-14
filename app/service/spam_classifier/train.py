"""
KoELECTRA 기반 범용 텍스트 분류 학습 파이프라인

이 코드는 생성 모델이 아닌 분류 모델 학습용임
EXAONE은 verdict/설명 생성 전용으로 분리됨

지원 기능:
- 베이스 모델: monologg/koelectra-small-v3-discriminator
- 학습 모드: Full Finetuning (기본) / LoRA (옵션)
- 다태스크: spam, sentiment 등
- 클래스 불균형 대응
- 평가 지표: accuracy, macro_f1, class별 precision/recall

실행 예시:
python train.py --task spam --mode full
python train.py --task sentiment --mode lora --lora_r 16
"""
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

# 프로젝트 루트를 Python 경로에 추가
_projects_root = Path(__file__).parent.parent.parent.parent
if str(_projects_root) not in sys.path:
    sys.path.insert(0, str(_projects_root))

try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
    )
    from datasets import Dataset
except ImportError as e:
    print(f"오류: 필요한 패키지가 설치되지 않았습니다: {e}")
    print("pip install transformers torch peft datasets scikit-learn 를 실행하세요.")
    sys.exit(1)

# 로컬 모듈 import
from app.service.spam_classifier.load_model import load_koelectra_model


def load_classification_datasets(dataset_dir: Path) -> tuple[Dataset, Dataset]:
    """분류용 Dataset 객체를 로드하고 검증합니다.

    Args:
        dataset_dir: Dataset이 저장된 디렉토리 경로

    Returns:
        (train_dataset, val_dataset) 튜플

    Raises:
        FileNotFoundError: Dataset 파일을 찾을 수 없을 때
        ValueError: 필수 컬럼이 없을 때
    """
    train_dataset_path = dataset_dir / "train_dataset"
    val_dataset_path = dataset_dir / "val_dataset"

    if not train_dataset_path.exists():
        raise FileNotFoundError(f"Train Dataset을 찾을 수 없습니다: {train_dataset_path}")
    if not val_dataset_path.exists():
        raise FileNotFoundError(f"Validation Dataset을 찾을 수 없습니다: {val_dataset_path}")

    train_dataset = Dataset.load_from_disk(str(train_dataset_path))
    val_dataset = Dataset.load_from_disk(str(val_dataset_path))

    # 필수 컬럼 검증
    required_columns = {"text", "label"}
    for name, dataset in [("train", train_dataset), ("val", val_dataset)]:
        missing_cols = required_columns - set(dataset.column_names)
        if missing_cols:
            raise ValueError(
                f"{name} dataset에 필수 컬럼이 없습니다: {missing_cols}\n"
                f"현재 컬럼: {dataset.column_names}\n"
                f"필수 컬럼: text (str), label (int)"
            )

    return train_dataset, val_dataset


def setup_classification_lora(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
    verbose: bool = True,
):
    """분류용 LoRA 어댑터를 모델에 추가합니다.

    Args:
        model: 베이스 분류 모델 (AutoModelForSequenceClassification)
        lora_r: LoRA rank (기본값: 8, 분류 태스크에 적합)
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: LoRA를 적용할 모듈 목록 (None이면 KoELECTRA 기본값)
        verbose: 상세 정보 출력 여부

    Returns:
        PEFT 모델 (LoRA 어댑터가 추가된 분류 모델)
    """
    if verbose:
        print("\n" + "=" * 50)
        print("분류용 LoRA 어댑터 설정")
        print("=" * 50)

    # KoELECTRA 분류 모델용 타겟 모듈
    if target_modules is None:
        target_modules = ["query", "key", "value", "dense"]
        if verbose:
            print(f"타겟 모듈 (KoELECTRA 기본값): {target_modules}")

    if verbose:
        print(f"LoRA 설정:")
        print(f"  - Rank (r): {lora_r}")
        print(f"  - Alpha: {lora_alpha}")
        print(f"  - Dropout: {lora_dropout}")
        print(f"  - Task Type: SEQ_CLS (분류)")

    # 분류용 LoRA 설정 (TaskType.SEQ_CLS 사용)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,  # 분류 태스크
    )

    # LoRA 어댑터 추가
    peft_model = get_peft_model(model, lora_config)

    if verbose:
        # 학습 가능한 파라미터 수 출력
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        print(f"\n[OK] LoRA 어댑터 추가 완료")
        print(f"  - 학습 가능 파라미터: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"  - 총 파라미터: {total_params:,}")
        print("=" * 50)

    return peft_model


def compute_classification_metrics(task: str = "spam"):
    """분류 태스크별 평가 지표 계산 함수를 생성합니다.

    Args:
        task: 태스크 이름 ("spam", "sentiment" 등)

    Returns:
        compute_metrics 함수
    """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # 기본 지표
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average='macro')

        # 클래스별 precision, recall, f1
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }

        # 클래스별 지표 추가
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            if i < len(precision):
                metrics[f"class_{label}_precision"] = precision[i]
                metrics[f"class_{label}_recall"] = recall[i]
                metrics[f"class_{label}_f1"] = f1[i]

        # 태스크별 특수 지표
        if task == "spam" and len(unique_labels) >= 2:
            # 스팸 태스크: 클래스 1을 스팸으로 가정
            spam_class = 1 if 1 in unique_labels else max(unique_labels)
            spam_idx = list(unique_labels).index(spam_class) if spam_class in unique_labels else -1

            if spam_idx >= 0 and spam_idx < len(recall):
                metrics["spam_recall"] = recall[spam_idx]
                metrics["spam_f1"] = f1[spam_idx]

        return metrics

    return compute_metrics


class WeightedTrainer(Trainer):
    """클래스 불균형 대응을 위한 가중치 적용 Trainer."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        """가중치가 적용된 손실 함수를 계산합니다."""
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if self.class_weights is not None:
            # 가중치 적용된 CrossEntropyLoss
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
            loss = loss_fct(outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            # 기본 손실 함수 사용
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def train_text_classifier(
    task: str = "spam",
    mode: str = "full",
    model_dir: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    *,
    # LoRA 설정 (mode="lora"일 때만 사용)
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
    # 학습 하이퍼파라미터
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    logging_steps: int = 50,
    save_steps: int = 500,
    save_total_limit: int = 2,
    max_seq_length: int = 256,
    fp16: bool = None,  # GPU에서만 자동 활성화
    # 클래스 불균형 대응
    use_class_weights: bool = False,
    # 평가 설정
    metric_for_best_model: str = None,  # 태스크별 자동 설정
    verbose: bool = True,
):
    """KoELECTRA 기반 텍스트 분류 모델을 학습합니다.

    Args:
        task: 태스크 이름 ("spam", "sentiment" 등)
        mode: 학습 모드 ("full" 또는 "lora")
        model_dir: 모델 디렉토리 경로 (None이면 기본 경로 사용)
        dataset_dir: Dataset이 저장된 디렉토리 경로
        output_dir: 출력 디렉토리 경로 (None이면 자동 생성)
        lora_r: LoRA rank (mode="lora"일 때만 사용)
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: LoRA를 적용할 모듈 목록
        num_train_epochs: 학습 에포크 수
        per_device_train_batch_size: 디바이스당 배치 크기
        gradient_accumulation_steps: 그래디언트 누적 스텝 수
        learning_rate: 학습률
        warmup_steps: 워밍업 스텝 수
        logging_steps: 로깅 스텝 간격
        save_steps: 저장 스텝 간격
        save_total_limit: 유지할 체크포인트 수
        max_seq_length: 최대 시퀀스 길이 (분류용으로 256 권장)
        fp16: FP16 사용 여부 (None이면 GPU에서 자동 활성화)
        use_class_weights: 클래스 불균형 대응 여부
        metric_for_best_model: 최적 모델 선정 기준 (None이면 태스크별 자동 설정)
        verbose: 상세 정보 출력 여부

    Returns:
        학습된 모델 (Full finetuning 또는 PEFT 모델)
    """
    # 입력 검증
    if mode not in ["full", "lora"]:
        raise ValueError(f"지원하지 않는 학습 모드: {mode}. 'full' 또는 'lora'를 사용하세요.")

    # 기본 경로 설정
    if dataset_dir is None:
        dataset_dir = Path(__file__).parent.parent.parent / "data" / f"{task}_processed"
    else:
        dataset_dir = Path(dataset_dir)

    if output_dir is None:
        # 자동 출력 경로 생성: models/{task}/{mode}/run_{timestamp}/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = (
            Path(__file__).parent.parent.parent / "model" / task / mode / f"run_{timestamp}"
        )
    else:
        output_dir = Path(output_dir)

    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if fp16 is None:
        fp16 = device == "cuda"  # GPU에서만 fp16 활성화

    # 태스크별 기본 metric 설정
    if metric_for_best_model is None:
        if task == "spam":
            metric_for_best_model = "spam_recall"
        else:
            metric_for_best_model = "macro_f1"

    if verbose:
        print("=" * 60)
        print(f"KoELECTRA {task.upper()} 분류 모델 학습")
        print("=" * 60)
        print(f"태스크: {task}")
        print(f"학습 모드: {mode}")
        print(f"디바이스: {device}")
        print(f"Dataset 디렉토리: {dataset_dir}")
        print(f"출력 디렉토리: {output_dir}")

    # 1. Dataset 로드 및 검증
    if verbose:
        print("\n[1/6] Dataset 로드 및 검증 중...")
    train_dataset, val_dataset = load_classification_datasets(dataset_dir)

    # 라벨 분포 분석
    train_labels = train_dataset["label"]
    unique_labels = sorted(set(train_labels))
    num_labels = len(unique_labels)

    label_counts = {label: train_labels.count(label) for label in unique_labels}

    if verbose:
        print(f"  Train Dataset: {len(train_dataset)}개 샘플")
        print(f"  Validation Dataset: {len(val_dataset)}개 샘플")
        print(f"  컬럼: {train_dataset.column_names}")
        print(f"  클래스 수: {num_labels}")
        print(f"  라벨 분포:")
        for label, count in label_counts.items():
            print(f"    클래스 {label}: {count}개 ({100*count/len(train_dataset):.1f}%)")

    # 2. 모델 및 토크나이저 로드
    if verbose:
        print(f"\n[2/6] KoELECTRA 분류 모델 로드 중... (클래스 수: {num_labels})")

    # 분류용 모델 로드
    if model_dir is None:
        model_name = "monologg/koelectra-small-v3-discriminator"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        # 로컬 모델 사용 (기존 load_koelectra_model 함수 활용 불가, 분류용이 아니므로)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            num_labels=num_labels,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

    if verbose:
        print(f"  모델 로드 완료: {type(model).__name__}")
        print(f"  토크나이저 로드 완료")

    # 3. LoRA 설정 (mode="lora"인 경우)
    if mode == "lora":
        if verbose:
            print(f"\n[3/6] LoRA 어댑터 설정 중...")
        model = setup_classification_lora(
            model=model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            verbose=verbose,
        )
    elif verbose:
        print(f"\n[3/6] Full Finetuning 모드 (LoRA 사용 안함)")

    # 4. 데이터 전처리
    if verbose:
        print(f"\n[4/6] 데이터 토큰화 중... (최대 길이: {max_seq_length})")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # DataCollator에서 동적 패딩
            max_length=max_seq_length,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # 토큰화 후 필요한 컬럼만 유지
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # 5. 클래스 가중치 계산 (옵션)
    class_weights = None
    if use_class_weights:
        if verbose:
            print(f"\n[5/6] 클래스 불균형 대응: 가중치 계산 중...")

        weights = compute_class_weight(
            "balanced",
            classes=np.array(unique_labels),
            y=np.array(train_labels)
        )
        class_weights = torch.FloatTensor(weights)

        if verbose:
            print(f"  클래스 가중치: {dict(zip(unique_labels, weights))}")
    elif verbose:
        print(f"\n[5/6] 클래스 가중치 사용 안함")

    # 6. 학습 설정
    if verbose:
        print(f"\n[6/6] 학습 설정 및 시작...")
        print(f"  - 에포크: {num_train_epochs}")
        print(f"  - 배치 크기: {per_device_train_batch_size}")
        print(f"  - 학습률: {learning_rate}")
        print(f"  - FP16: {fp16}")
        print(f"  - 최적 모델 기준: {metric_for_best_model}")

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        eval_strategy="steps",  # 최신 transformers에서는 eval_strategy 사용
        eval_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,  # recall, f1은 클수록 좋음
        fp16=fp16,
        bf16=False,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )

    # 데이터 콜레이터
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 평가 지표 함수
    compute_metrics = compute_classification_metrics(task=task)

    # Trainer 생성 (가중치 적용 여부에 따라)
    if use_class_weights:
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,  # tokenizer 대신 processing_class 사용
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,  # tokenizer 대신 processing_class 사용
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    # 학습 실행
    if verbose:
        print("\n학습을 시작합니다...")

    trainer.train()

    # 최종 모델 및 메트릭 저장
    if verbose:
        print("\n최종 모델 저장 중...")

    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))

    # 최종 평가 및 메트릭 저장
    final_metrics = trainer.evaluate()
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    if verbose:
        print("\n" + "=" * 60)
        print("[SUCCESS] 학습 완료!")
        print("=" * 60)
        print(f"모델 저장 위치: {output_dir}")
        print(f"  - config.json: 모델 설정")
        print(f"  - pytorch_model.bin: 모델 가중치")
        print(f"  - tokenizer 파일들")
        print(f"  - metrics.json: 평가 지표")
        print(f"\n최종 성능:")
        for key, value in final_metrics.items():
            if not key.startswith("eval_"):
                continue
            metric_name = key.replace("eval_", "")
            print(f"  {metric_name}: {value:.4f}")
        print("=" * 60)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="KoELECTRA 기반 범용 텍스트 분류 모델 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 스팸 분류 Full Finetuning
  python train.py --task spam --mode full

  # 감정 분석 LoRA 학습
  python train.py --task sentiment --mode lora --lora_r 16

  # 클래스 불균형 대응
  python train.py --task spam --mode full --use_class_weights
        """
    )

    # 필수 인자
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="태스크 이름 (spam, sentiment 등)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "lora"],
        default="full",
        help="학습 모드: full (전체 파인튜닝) 또는 lora (LoRA 어댑터)"
    )

    # 경로 설정
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="모델 디렉토리 경로 (기본값: HuggingFace Hub에서 다운로드)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Dataset 디렉토리 경로 (기본값: app/data/{task}_processed)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="출력 디렉토리 경로 (기본값: 자동 생성)"
    )

    # LoRA 설정
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank (기본값: 8, 분류 태스크 최적화)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (기본값: 16)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (기본값: 0.1)"
    )

    # 학습 하이퍼파라미터
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="학습 에포크 수 (기본값: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="배치 크기 (기본값: 16)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="학습률 (기본값: 2e-5, 분류 태스크 최적화)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help="최대 시퀀스 길이 (기본값: 256)"
    )

    # 클래스 불균형 대응
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="클래스 불균형 대응을 위한 가중치 사용"
    )

    # 평가 설정
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default=None,
        help="최적 모델 선정 기준 (기본값: 태스크별 자동 설정)"
    )

    args = parser.parse_args()

    # 경로 변환
    model_dir = Path(args.model_dir) if args.model_dir else None
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        model = train_text_classifier(
            task=args.task,
            mode=args.mode,
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            use_class_weights=args.use_class_weights,
            metric_for_best_model=args.metric_for_best_model,
            verbose=True,
        )

        print(f"\n{args.task} 분류 모델 학습이 성공적으로 완료되었습니다!")
        print(f"학습 모드: {args.mode}")
        print(f"모델 타입: {type(model).__name__}")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

