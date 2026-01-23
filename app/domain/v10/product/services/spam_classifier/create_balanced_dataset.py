"""
균형잡힌 테스트 데이터셋을 생성하는 스크립트
"""
from pathlib import Path
from datasets import Dataset
import random

def create_balanced_test_dataset():
    """스팸/정상 균형잡힌 테스트 데이터셋을 생성합니다."""

    # 기존 스팸 데이터 로드
    spam_dataset = Dataset.load_from_disk("app/data/spam_processed/train_dataset")

    # 스팸 데이터 일부만 사용 (테스트용)
    spam_samples = spam_dataset.select(range(min(1000, len(spam_dataset))))

    # 정상 메일 더미 데이터 생성
    normal_texts = [
        "안녕하세요. 회의 일정을 알려드립니다.",
        "프로젝트 진행 상황을 공유드립니다.",
        "오늘 점심 메뉴는 무엇인가요?",
        "보고서 검토 부탁드립니다.",
        "내일 미팅 시간을 변경하고 싶습니다.",
        "새로운 정책에 대해 안내드립니다.",
        "시스템 점검 일정을 알려드립니다.",
        "교육 프로그램 참가 신청 안내",
        "월간 실적 보고서입니다.",
        "고객 문의사항에 대한 답변입니다.",
        "팀 빌딩 행사 참여 여부를 확인해주세요.",
        "신제품 출시 일정을 공유합니다.",
        "휴가 신청서를 제출해주세요.",
        "예산 계획서를 검토해주시기 바랍니다.",
        "컨퍼런스 참가 신청 마감일이 다가옵니다.",
    ]

    # 정상 데이터 1000개 생성
    normal_data = []
    for i in range(1000):
        text = random.choice(normal_texts)
        # 약간의 변형 추가
        if i % 3 == 0:
            text = f"[공지] {text}"
        elif i % 3 == 1:
            text = f"{text} 감사합니다."

        normal_data.append({
            "text": text,
            "label": 0  # 정상
        })

    # 스팸 데이터 준비
    spam_data = []
    for sample in spam_samples:
        spam_data.append({
            "text": sample["text"],
            "label": 1  # 스팸
        })

    # 전체 데이터 합치기
    all_data = normal_data + spam_data
    random.shuffle(all_data)

    # 80:20 분할
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # Dataset 객체 생성
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    print(f"균형잡힌 데이터셋 생성 완료:")
    print(f"  Train: {len(train_dataset)}개 샘플")
    print(f"  Val: {len(val_dataset)}개 샘플")

    # 라벨 분포 확인
    train_labels = train_dataset["label"]
    label_counts = {0: train_labels.count(0), 1: train_labels.count(1)}

    print(f"  라벨 분포:")
    print(f"    클래스 0 (정상): {label_counts[0]}개 ({100*label_counts[0]/len(train_dataset):.1f}%)")
    print(f"    클래스 1 (스팸): {label_counts[1]}개 ({100*label_counts[1]/len(train_dataset):.1f}%)")

    # 저장
    output_dir = Path("app/data/spam_balanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset.save_to_disk(str(output_dir / "train_dataset"))
    val_dataset.save_to_disk(str(output_dir / "val_dataset"))

    print(f"\n저장 완료: {output_dir}")

if __name__ == "__main__":
    create_balanced_test_dataset()
