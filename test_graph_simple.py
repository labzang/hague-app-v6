#!/usr/bin/env python3
"""
간단한 그래프 테스트 (오류 처리 확인)
"""

def test_basic_functionality():
    """기본 기능 테스트"""
    try:
        print("=" * 50)
        print("그래프 기본 기능 테스트")
        print("=" * 50)

        # 그래프 import
        from app.graph import run_once, graph

        print(f"그래프 상태: {'초기화됨' if graph is not None else '초기화 실패'}")

        # 간단한 테스트
        print("\n[테스트 1] 간단한 질문")
        result = run_once("안녕하세요!")
        print(f"응답: {result}")

        print("\n[테스트 2] 빈 입력")
        result = run_once("")
        print(f"응답: {result}")

        print("\n[테스트 3] None 입력")
        result = run_once(None)  # type: ignore
        print(f"응답: {result}")

        print("\n" + "=" * 50)
        print("기본 테스트 완료!")
        print("=" * 50)

    except Exception as e:
        print(f"[오류] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_functionality()
