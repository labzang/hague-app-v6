"""
KoElectra 모델을 사용하여 사용자 요구사항을 정책기반/규칙기반으로 분류하는 모델 훈련 스크립트.

라벨:
- 0: BLOCK (서비스 범위 밖)
- 1: POLICY_BASED (정책 기반, LLM 필요)
- 2: RULE_BASED (규칙 기반, DB 쿼리)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np


@dataclass
class TrainingConfig:
    """훈련 설정"""
    model_path: str = "artifacts/models--monologg--koelectra-small-v3-discriminator"
    train_data_path: str = "artfacts-training/koelectra_training_dataset.sft.jsonl"
    output_dir: str = "artfacts-training/koelectra_orchestrator_finetuned"
    num_labels: int = 3  # BLOCK(0), POLICY_BASED(1), RULE_BASED(2)
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    save_total_limit: int = 3
    logging_steps: int = 50
    seed: int = 42


class OrchestratorDataset(Dataset):
    """오케스트레이터 분류를 위한 데이터셋"""

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        """
        Args:
            data_path: JSONL 파일 경로
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        """JSONL 파일에서 데이터 로드"""
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    question = item["input"]["question"]
                    label = item["label"]
                    data.append({"text": question, "label": label})
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"데이터 로드 오류 (라인 건너뜀): {e}")
                    continue
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        # 토크나이징
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_metrics(eval_pred: Tuple) -> Dict[str, float]:
    """평가 메트릭 계산"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    # 클래스별 메트릭
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    label_names = ["BLOCK", "POLICY_BASED", "RULE_BASED"]
    per_class_metrics = {}
    for i, name in enumerate(label_names):
        per_class_metrics[f"{name}_precision"] = float(precision_per_class[i])
        per_class_metrics[f"{name}_recall"] = float(recall_per_class[i])
        per_class_metrics[f"{name}_f1"] = float(f1_per_class[i])

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **per_class_metrics,
    }


def load_model_and_tokenizer(model_path: str, num_labels: int = 3):
    """모델과 토크나이저 로드"""
    print(f"모델 로딩 중: {model_path}")

    # 모델 경로 확인
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {model_dir}")

    # 토크나이저 파일이 있는 스냅샷 찾기
    tokenizer_path = None
    model_snapshot_path = None

    snapshots_dir = model_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            # 토크나이저 파일(vocab.txt)이 있는 스냅샷 찾기
            for snapshot in snapshots:
                snapshot_path = snapshots_dir / snapshot
                vocab_file = snapshot_path / "vocab.txt"
                if vocab_file.exists():
                    tokenizer_path = str(snapshot_path)
                    print(f"토크나이저 스냅샷 발견: {tokenizer_path}")
                    break

            # 모델 파일이 있는 스냅샷 찾기 (pytorch_model.bin 또는 model.safetensors)
            for snapshot in snapshots:
                snapshot_path = snapshots_dir / snapshot
                if (snapshot_path / "pytorch_model.bin").exists() or \
                   (snapshot_path / "model.safetensors").exists():
                    model_snapshot_path = str(snapshot_path)
                    print(f"모델 스냅샷 발견: {model_snapshot_path}")
                    break

    # 토크나이저 경로 설정 (스냅샷에 없으면 원본 경로 사용)
    if tokenizer_path is None:
        tokenizer_path = str(model_dir)
        print(f"토크나이저를 원본 경로에서 로드: {tokenizer_path}")

    # 모델 경로 설정 (스냅샷에 없으면 원본 경로 사용)
    if model_snapshot_path is None:
        model_snapshot_path = str(model_dir)
        print(f"모델을 원본 경로에서 로드: {model_snapshot_path}")

    # 토크나이저 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
        )
    except Exception as e:
        print(f"토크나이저 로드 실패 ({tokenizer_path}): {e}")
        # 원본 모델 이름으로 시도
        print("원본 모델 이름으로 토크나이저 로드 시도...")
        tokenizer = AutoTokenizer.from_pretrained(
            "monologg/koelectra-small-v3-discriminator",
            local_files_only=False,
        )

    # 모델 로드 (Sequence Classification으로 변환)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_snapshot_path,
            num_labels=num_labels,
            local_files_only=True,
        )
    except Exception as e:
        print(f"모델 로드 실패 ({model_snapshot_path}): {e}")
        # 원본 모델 이름으로 시도
        print("원본 모델 이름으로 모델 로드 시도...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "monologg/koelectra-small-v3-discriminator",
            num_labels=num_labels,
            local_files_only=False,
        )

    # pad_token이 없으면 추가
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    print(f"모델 로딩 완료: {num_labels}개 클래스")
    return model, tokenizer


def split_dataset(dataset: OrchestratorDataset, train_ratio: float = 0.8) -> Tuple[Dataset, Dataset]:
    """데이터셋을 훈련/검증 세트로 분할"""
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    return train_dataset, val_dataset


def train(config: TrainingConfig):
    """모델 훈련 실행"""
    print("=" * 60)
    print("KoElectra 오케스트레이터 분류 모델 훈련 시작")
    print("=" * 60)

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # 모델과 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(config.model_path, config.num_labels)
    model.to(device)

    # 데이터셋 로드
    print(f"\n데이터셋 로딩 중: {config.train_data_path}")
    full_dataset = OrchestratorDataset(
        config.train_data_path,
        tokenizer,
        max_length=config.max_length,
    )
    print(f"전체 데이터 수: {len(full_dataset)}")

    # 라벨 분포 확인
    labels = [item["label"] for item in full_dataset.data]
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"라벨 분포: {label_counts}")

    # 훈련/검증 세트 분할
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8)
    print(f"훈련 세트: {len(train_dataset)}개")
    print(f"검증 세트: {len(val_dataset)}개")

    # 훈련 인자 설정
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        fp16=torch.cuda.is_available(),  # GPU 사용 시 fp16 활성화
        report_to="none",  # wandb/tensorboard 비활성화
    )

    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 훈련 시작
    print("\n" + "=" * 60)
    print("훈련 시작")
    print("=" * 60)
    train_result = trainer.train()

    # 최종 평가
    print("\n" + "=" * 60)
    print("최종 평가")
    print("=" * 60)
    eval_result = trainer.evaluate()
    print(f"\n최종 평가 결과:")
    for key, value in eval_result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 모델 저장
    print(f"\n모델 저장 중: {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    print("모델 저장 완료")

    # 훈련 결과 저장
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_loss": train_result.training_loss,
                "eval_metrics": eval_result,
                "label_distribution": label_counts,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("훈련 완료!")
    print("=" * 60)
    print(f"모델 저장 위치: {config.output_dir}")


if __name__ == "__main__":
    # 스크립트 위치 기준으로 프로젝트 루트 찾기
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == "artfacts-training" else script_dir

    # 훈련 설정
    config = TrainingConfig()

    # 경로를 절대 경로로 변환
    config.model_path = str(project_root / config.model_path)
    config.train_data_path = str(project_root / config.train_data_path)
    config.output_dir = str(project_root / config.output_dir)

    # 경로 확인
    if not Path(config.train_data_path).exists():
        raise FileNotFoundError(f"훈련 데이터 파일을 찾을 수 없습니다: {config.train_data_path}")

    # 훈련 실행
    train(config)

