# LangSmith 모니터링 설정 가이드

## 개요

LangSmith는 LangGraph 및 LangChain 실행을 추적하고 모니터링하는 플랫폼입니다.
이 프로젝트의 모든 Orchestrator는 LangSmith를 통해 실행 흐름을 추적할 수 있습니다.

## 설정 방법

### 1. LangSmith 계정 생성 및 API 키 발급

1. [LangSmith](https://smith.langchain.com)에 접속하여 계정을 생성합니다.
2. Settings → API Keys에서 API 키를 발급받습니다.

### 2. 환경 변수 설정

`.env` 파일에 다음 환경 변수를 추가합니다:

```bash
# LangSmith 모니터링 활성화
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=soccer-data-processing
```

**환경 변수 설명:**
- `LANGCHAIN_TRACING_V2`: LangSmith 추적 활성화 여부 (`true` 또는 `false`)
- `LANGSMITH_API_KEY`: LangSmith에서 발급받은 API 키
- `LANGCHAIN_PROJECT`: LangSmith에서 표시될 프로젝트 이름 (기본값: `soccer-data-processing`)

### 3. 패키지 설치

LangSmith 패키지가 이미 `requirements.txt`에 포함되어 있습니다:

```bash
pip install langsmith>=0.1.0
```

## 사용 방법

### 자동 추적

환경 변수가 설정되어 있으면 자동으로 LangSmith 추적이 활성화됩니다.

각 Orchestrator의 `process_*` 메서드가 호출될 때:
- LangGraph 실행이 자동으로 추적됩니다
- 각 노드의 실행 시간, 입력/출력이 기록됩니다
- 에러 발생 시 상세한 스택 트레이스가 기록됩니다

### 추적되는 정보

- **그래프 실행**: 전체 LangGraph 실행 흐름
- **노드 실행**: 각 노드(validate, determine_strategy, policy_process, rule_process, finalize)의 실행
- **상태 변화**: StateGraph의 상태 변화 추적
- **메타데이터**:
  - 도메인 (player, team, stadium, schedule)
  - 처리된 항목 수
  - 사용된 전략 (policy/rule)
- **태그**:
  - `langgraph`
  - `soccer`
  - `data-processing`
  - `{domain}-processing` (예: `player-processing`)

## LangSmith 대시보드에서 확인

1. [LangSmith 대시보드](https://smith.langchain.com)에 로그인합니다.
2. 프로젝트 선택: `LANGCHAIN_PROJECT`에 설정한 프로젝트 이름을 선택합니다.
3. 실행 추적 확인:
   - 각 API 요청마다 하나의 "Run"이 생성됩니다
   - 그래프 구조와 노드 실행 순서를 시각적으로 확인할 수 있습니다
   - 각 노드의 입력/출력, 실행 시간, 에러 정보를 확인할 수 있습니다

## 비활성화

LangSmith 추적을 비활성화하려면:

```bash
# .env 파일에서 주석 처리하거나 삭제
# LANGCHAIN_TRACING_V2=false
# 또는 환경 변수 제거
```

또는 환경 변수를 설정하지 않으면 자동으로 비활성화됩니다.

## 코드 구조

### 설정 파일
- `app/core/langsmith_config.py`: LangSmith 설정 관리

### Orchestrator 통합
모든 Orchestrator (`player_orchestrator.py`, `team_orchestrator.py`, `stadium_orchestrator.py`, `schedule_orchestrator.py`)에 LangSmith 추적이 통합되어 있습니다.

### 사용 예시

```python
from app.core.langsmith_config import get_langsmith_config

# LangSmith config 가져오기
langsmith_config = get_langsmith_config()

# LangGraph 실행 시 config 전달
final_state = await self.graph.ainvoke(
    initial_state,
    config=langsmith_config
)
```

## 문제 해결

### 추적이 되지 않는 경우

1. 환경 변수 확인:
   ```bash
   echo $LANGSMITH_API_KEY
   echo $LANGCHAIN_TRACING_V2
   ```

2. 로그 확인:
   - `[LangSmith] 추적 활성화: 프로젝트=...` 메시지가 출력되는지 확인
   - `[LangSmith] 추적 비활성화` 메시지가 출력되면 환경 변수를 확인하세요

3. API 키 유효성 확인:
   - LangSmith 대시보드에서 API 키가 활성화되어 있는지 확인

### 성능 영향

LangSmith 추적은 비동기로 실행되므로 성능에 미치는 영향이 최소화됩니다.
하지만 대량의 데이터를 처리할 때는 약간의 오버헤드가 있을 수 있습니다.

## 추가 리소스

- [LangSmith 공식 문서](https://docs.smith.langchain.com/)
- [LangGraph 모니터링 가이드](https://langchain-ai.github.io/langgraph/how-tos/monitoring/)

