"""CSV 파일을 JSONL 형식으로 변환하는 유틸리티 모듈."""
import csv
import json
import os
from pathlib import Path


def convert_csv_to_jsonl(csv_file_path: str, jsonl_file_path: str | None = None) -> str:
    """CSV 파일을 JSONL 형식으로 변환합니다.

    Args:
        csv_file_path: 변환할 CSV 파일 경로
        jsonl_file_path: 출력할 JSONL 파일 경로 (None이면 자동 생성)

    Returns:
        생성된 JSONL 파일 경로

    Raises:
        FileNotFoundError: CSV 파일이 존재하지 않을 때
        ValueError: CSV 파일 형식이 올바르지 않을 때
    """
    csv_path = Path(csv_file_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_file_path}")

    # 출력 파일 경로가 지정되지 않으면 자동 생성 (확장자만 변경)
    if jsonl_file_path is None:
        jsonl_file_path = csv_path.with_suffix(".jsonl")
    else:
        jsonl_file_path = Path(jsonl_file_path)

    # CSV 파일 읽기 및 JSONL 파일 쓰기
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
            # CSV Reader 생성 (따옴표 처리 자동)
            reader = csv.DictReader(csv_file)

            # JSONL 파일 쓰기
            with open(jsonl_file_path, "w", encoding="utf-8") as jsonl_file:
                row_count = 0
                for row in reader:
                    # 각 행을 JSON 문자열로 변환하여 한 줄씩 쓰기
                    json_line = json.dumps(row, ensure_ascii=False)
                    jsonl_file.write(json_line + "\n")
                    row_count += 1

                print(f"변환 완료: {row_count}개 행이 변환되었습니다.")
                print(f"입력 파일: {csv_path}")
                print(f"출력 파일: {jsonl_file_path}")

        return str(jsonl_file_path)

    except csv.Error as e:
        raise ValueError(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
    except json.JSONEncodeError as e:
        raise ValueError(f"JSON 변환 중 오류가 발생했습니다: {e}")
    except Exception as e:
        raise RuntimeError(f"파일 변환 중 예상치 못한 오류가 발생했습니다: {e}")


def convert_all_csv_in_directory(directory_path: str) -> list[str]:
    """디렉토리 내의 모든 CSV 파일을 JSONL로 변환합니다.

    Args:
        directory_path: 검색할 디렉토리 경로

    Returns:
        변환된 JSONL 파일 경로 목록
    """
    dir_path = Path(directory_path)

    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"디렉토리를 찾을 수 없습니다: {directory_path}")

    csv_files = list(dir_path.glob("*.csv"))
    converted_files = []

    for csv_file in csv_files:
        try:
            jsonl_path = convert_csv_to_jsonl(str(csv_file))
            converted_files.append(jsonl_path)
        except Exception as e:
            print(f"오류: {csv_file.name} 변환 실패 - {e}")

    return converted_files


if __name__ == "__main__":
    import sys

    # data 디렉토리 경로
    data_dir = Path(__file__).parent.parent / "data"

    if not data_dir.exists():
        print(f"오류: data 디렉토리를 찾을 수 없습니다: {data_dir}")
        sys.exit(1)

    # 명령줄 인자가 있으면 특정 CSV 파일 변환
    if len(sys.argv) > 1:
        csv_file = Path(sys.argv[1])
        if not csv_file.exists():
            print(f"오류: 파일을 찾을 수 없습니다: {csv_file}")
            sys.exit(1)
        try:
            convert_csv_to_jsonl(str(csv_file))
        except Exception as e:
            print(f"오류: {e}")
            sys.exit(1)
    else:
        # data 디렉토리 내의 모든 CSV 파일 변환
        print(f"'{data_dir}' 디렉토리 내의 CSV 파일을 변환합니다...")
        try:
            converted = convert_all_csv_in_directory(str(data_dir))
            if converted:
                print(f"\n총 {len(converted)}개 파일이 변환되었습니다.")
            else:
                print("변환할 CSV 파일이 없습니다.")
        except Exception as e:
            print(f"오류: {e}")
            sys.exit(1)

