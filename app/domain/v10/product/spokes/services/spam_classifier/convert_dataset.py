"""
SFT 형식의 데이터셋을 분류용으로 변환하는 스크립트

기존: {'instruction', 'input', 'output', 'text'}
변환: {'text', 'label'}
"""
import json
from pathlib import Path
from datasets import Dataset

def convert_sft_to_classification(dataset_dir: Path, output_dir: Path):
    """SFT 데이터셋을 분류용으로 변환합니다."""

    # 원본 데이터셋 로드
    train_dataset = Dataset.load_from_disk(str(dataset_dir / "train_dataset"))
    val_dataset = Dataset.load_from_disk(str(dataset_dir / "val_dataset"))

    print(f"원본 데이터셋 로드 완료:")
    print(f"  Train: {len(train_dataset)}개 샘플")
    print(f"  Val: {len(val_dataset)}개 샘플")
    print(f"  컬럼: {train_dataset.column_names}")

    def convert_sample(example):
        """개별 샘플을 분류용으로 변환합니다."""
        # input의 subject를 텍스트로 사용
        if isinstance(example['input'], dict):
            text = example['input'].get('subject', '')
        else:
            # input이 문자열인 경우 JSON 파싱 시도
            try:
                input_data = json.loads(example['input'])
                text = input_data.get('subject', '')
            except:
                text = str(example['input'])

        # output의 action을 라벨로 변환
        if isinstance(example['output'], dict):
            action = example['output'].get('action', 'ALLOW')
        else:
            # output이 문자열인 경우 JSON 파싱 시도
            try:
                output_data = json.loads(example['output'])
                action = output_data.get('action', 'ALLOW')
            except:
                action = 'ALLOW'

        # 라벨 매핑: BLOCK=1 (스팸), ALLOW=0 (정상)
        label = 1 if action == 'BLOCK' else 0

        return {
            'text': text,
            'label': label
        }

    # 데이터셋 변환
    print("\n데이터셋 변환 중...")
    train_converted = train_dataset.map(convert_sample)
    val_converted = val_dataset.map(convert_sample)

    # 필요한 컬럼만 선택
    train_converted = train_converted.select_columns(['text', 'label'])
    val_converted = val_converted.select_columns(['text', 'label'])

    # 라벨 분포 확인
    train_labels = train_converted['label']
    label_counts = {0: train_labels.count(0), 1: train_labels.count(1)}

    print(f"\n변환된 데이터셋:")
    print(f"  Train: {len(train_converted)}개 샘플")
    print(f"  Val: {len(val_converted)}개 샘플")
    print(f"  컬럼: {train_converted.column_names}")
    print(f"  라벨 분포:")
    print(f"    클래스 0 (정상): {label_counts[0]}개 ({100*label_counts[0]/len(train_converted):.1f}%)")
    print(f"    클래스 1 (스팸): {label_counts[1]}개 ({100*label_counts[1]/len(train_converted):.1f}%)")

    # 샘플 확인
    print(f"\n샘플 데이터:")
    for i in range(min(3, len(train_converted))):
        sample = train_converted[i]
        print(f"  [{i}] 텍스트: {sample['text'][:100]}...")
        print(f"      라벨: {sample['label']} ({'스팸' if sample['label'] == 1 else '정상'})")

    # 저장
    output_dir.mkdir(parents=True, exist_ok=True)

    train_output_path = output_dir / "train_dataset"
    val_output_path = output_dir / "val_dataset"

    train_converted.save_to_disk(str(train_output_path))
    val_converted.save_to_disk(str(val_output_path))

    print(f"\n변환 완료!")
    print(f"  저장 위치: {output_dir}")
    print(f"  - {train_output_path}")
    print(f"  - {val_output_path}")

if __name__ == "__main__":
    # 경로 설정
    dataset_dir = Path("app/data/spam_agent_processed")
    output_dir = Path("app/data/spam_processed")

    convert_sft_to_classification(dataset_dir, output_dir)
