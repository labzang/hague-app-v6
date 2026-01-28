"""축구 데이터를 활용한 KoElectra 학습 데이터 생성 스크립트."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# 데이터 로드
def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """JSONL 파일을 로드합니다."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# 데이터 로드
players = load_jsonl(Path("app/data/soccer/players.jsonl"))
teams = load_jsonl(Path("app/data/soccer/teams.jsonl"))
schedules = load_jsonl(Path("app/data/soccer/schedules.jsonl"))
stadiums = load_jsonl(Path("app/data/soccer/stadiums.jsonl"))

# 팀 이름 매핑 (team_id -> team_name)
team_map = {team['id']: team for team in teams}
stadium_map = {stadium['id']: stadium for stadium in stadiums}

# 선수 필터링 (유효한 데이터만)
valid_players = [p for p in players if p.get('player_name') and p.get('team_id')]
valid_schedules = [s for s in schedules if s.get('home_score') is not None and s.get('away_score') is not None]

training_data = []

# ===== LABEL 2: RULE_BASED (DB_QUERY) - 150개 =====
rule_based_questions = []

# 선수 관련 질문 (50개)
for i in range(50):
    player = random.choice(valid_players)
    team = team_map.get(player.get('team_id', 0), {})
    team_name = team.get('team_name', '')

    question_templates = [
        f"{player['player_name']} 등번호가 몇 번이야?",
        f"{player['player_name']} 포지션이 뭐야?",
        f"{player['player_name']} 키가 몇 cm야?",
        f"{player['player_name']} 몸무게가 몇 kg이야?",
        f"{player['player_name']} 생년월일이 언제야?",
        f"{player['player_name']} 국적이 어디야?",
        f"{player['player_name']} 소속 팀이 어디야?",
        f"{team_name}에 {player['player_name']} 선수가 있어?",
        f"{player['player_name']} 입단년도가 언제야?",
        f"{player['player_name']} 별명이 뭐야?" if player.get('nickname') else f"{player['player_name']} 정보 알려줘",
    ]
    question = random.choice(question_templates)
    rule_based_questions.append({
        "input": {"question": question, "intent": "DB_QUERY"},
        "output": {"action": "RULE_BASED", "reason": "명확한 데이터 조회 요청으로 규칙 기반 처리 가능", "confidence": 0.95},
        "label": 2
    })

# 팀 관련 질문 (30개)
for i in range(30):
    team = random.choice(teams)
    stadium = stadium_map.get(team.get('stadium_id', 0), {})

    question_templates = [
        f"{team['team_name']} 창단년도가 언제야?",
        f"{team['team_name']} 홈 경기장이 어디야?",
        f"{team['region_name']} 팀 주소가 어디야?",
        f"{team['team_name']} 전화번호가 뭐야?",
        f"{team['team_name']} 홈페이지 주소 알려줘",
        f"{team['team_name']} 팀 코드가 뭐야?",
        f"{stadium.get('statdium_name', '경기장')} 좌석 수가 몇 개야?",
        f"{team['team_name']} 정보 조회",
        f"{team['region_name']} 지역 팀 이름이 뭐야?",
        f"{team['team_name']} 연락처 알려줘",
    ]
    question = random.choice(question_templates)
    rule_based_questions.append({
        "input": {"question": question, "intent": "DB_QUERY"},
        "output": {"action": "RULE_BASED", "reason": "명확한 데이터 조회 요청으로 규칙 기반 처리 가능", "confidence": 0.95},
        "label": 2
    })

# 경기 일정 관련 질문 (40개)
for i in range(40):
    schedule = random.choice(valid_schedules)
    home_team = team_map.get(schedule.get('hometeam_id', 0), {})
    away_team = team_map.get(schedule.get('awayteam_id', 0), {})
    stadium = stadium_map.get(schedule.get('stadium_id', 0), {})

    question_templates = [
        f"{home_team.get('team_name', '')} vs {away_team.get('team_name', '')} 경기 결과 알려줘",
        f"{schedule.get('sche_date', '')} 경기 일정 보여줘",
        f"{home_team.get('team_name', '')} 홈 경기 일정 조회",
        f"{schedule.get('sche_date', '')}에 {home_team.get('team_name', '')} 경기 있나?",
        f"{home_team.get('team_name', '')} {schedule.get('home_score', 0)}:{schedule.get('away_score', 0)} {away_team.get('team_name', '')} 경기 정보",
        f"{stadium.get('statdium_name', '경기장')}에서 열린 경기 결과",
        f"{home_team.get('team_name', '')} 최근 경기 결과",
        f"{schedule.get('sche_date', '')} 경기 스코어",
    ]
    question = random.choice(question_templates)
    rule_based_questions.append({
        "input": {"question": question, "intent": "DB_QUERY"},
        "output": {"action": "RULE_BASED", "reason": "명확한 데이터 조회 요청으로 규칙 기반 처리 가능", "confidence": 0.95},
        "label": 2
    })

# 경기장 관련 질문 (30개)
for i in range(30):
    stadium = random.choice(stadiums)
    team = next((t for t in teams if t.get('stadium_id') == stadium['id']), {})

    question_templates = [
        f"{stadium.get('statdium_name', '경기장')} 좌석 수가 몇 개야?",
        f"{stadium.get('statdium_name', '경기장')} 주소가 어디야?",
        f"{stadium.get('statdium_name', '경기장')} 전화번호 알려줘",
        f"{stadium.get('statdium_name', '경기장')} 홈팀이 어디야?",
        f"{stadium.get('statdium_name', '경기장')} 정보 조회",
        f"{team.get('team_name', '')} 홈 경기장 이름이 뭐야?",
        f"{stadium.get('statdium_name', '경기장')} 위치 알려줘",
    ]
    question = random.choice(question_templates)
    rule_based_questions.append({
        "input": {"question": question, "intent": "DB_QUERY"},
        "output": {"action": "RULE_BASED", "reason": "명확한 데이터 조회 요청으로 규칙 기반 처리 가능", "confidence": 0.95},
        "label": 2
    })

training_data.extend(rule_based_questions[:150])

# ===== LABEL 1: POLICY_BASED (INFERENCE/ADVICE/EXPLAIN) - 90개 =====
policy_based_questions = []

# 추론 질문 (30개)
inference_templates = [
    "K리그 우승 가능성 높은 팀 순위 알려줘",
    "이번 시즌 가장 강한 팀은 어디야?",
    "다음 경기에서 승리 가능성 높은 팀은?",
    "선수 영입이 필요한 팀은 어디야?",
    "경기장 규모가 큰 팀 순위 알려줘",
    "창단 오래된 팀들이 더 강한가요?",
    "포지션별로 가장 많은 선수 보유 팀은?",
    "외국인 선수가 많은 팀은 어디야?",
    "가장 높은 선수는 누구야?",
    "가장 무거운 선수는 누구야?",
    "평균 키가 높은 팀은 어디야?",
    "최근 경기에서 가장 많이 득점한 팀은?",
    "홈 경기 승률이 높은 팀은 어디야?",
    "원정 경기에서 강한 팀은?",
    "경기장이 큰 팀이 유리한가요?",
]

for i in range(30):
    question = random.choice(inference_templates)
    policy_based_questions.append({
        "input": {"question": question, "intent": "INFERENCE"},
        "output": {"action": "POLICY_BASED", "reason": "복잡한 추론이 필요한 요청으로 정책 기반 처리 필요 (LLM 사용)", "confidence": 0.9},
        "label": 1
    })

# 조언 질문 (30개)
advice_templates = [
    "축구를 잘하려면 어떻게 연습해야 해?",
    "선수 영입 전략을 알려줘",
    "팀 전술을 어떻게 짜야 할까?",
    "경기장 운영 방법을 알려줘",
    "선수 관리 어떻게 하는 게 좋을까?",
    "팀 분위기 좋게 만드는 방법",
    "경기 전 준비는 어떻게 해야 해?",
    "부상 예방 방법 알려줘",
    "체력 향상 방법",
    "기술 향상 연습법",
    "팀워크 향상 방법",
    "경기 중 집중력 유지 방법",
    "선수 영입 시 고려사항",
    "경기장 시설 개선 방안",
    "팬들과 소통하는 방법",
]

for i in range(30):
    question = random.choice(advice_templates)
    policy_based_questions.append({
        "input": {"question": question, "intent": "ADVICE"},
        "output": {"action": "POLICY_BASED", "reason": "복잡한 추론이 필요한 요청으로 정책 기반 처리 필요 (LLM 사용)", "confidence": 0.9},
        "label": 1
    })

# 설명 질문 (30개)
explain_templates = [
    "선수들 연봉협상에서 가장 유리한 팀은 어디야?",
    "경기장 크기가 팀 성적에 영향을 주나요?",
    "외국인 선수가 팀에 미치는 영향은?",
    "홈 경기장 이점이 얼마나 큰가요?",
    "포지션별 역할을 설명해줘",
    "팀 전술의 중요성은?",
    "선수 영입이 팀 성적에 미치는 영향",
    "경기 일정이 팀 컨디션에 주는 영향",
    "팀 창단년도와 성적의 상관관계",
    "경기장 좌석 수와 팀 인기의 관계",
    "지역 연고가 팀에 미치는 영향",
    "선수 나이와 경기력의 관계",
    "포지션별 평균 키와 몸무게의 의미",
    "경기 결과가 다음 경기에 주는 영향",
    "팀 홈페이지가 팬 유치에 미치는 영향",
]

for i in range(30):
    question = random.choice(explain_templates)
    policy_based_questions.append({
        "input": {"question": question, "intent": "EXPLAIN"},
        "output": {"action": "POLICY_BASED", "reason": "복잡한 추론이 필요한 요청으로 정책 기반 처리 필요 (LLM 사용)", "confidence": 0.95},
        "label": 1
    })

training_data.extend(policy_based_questions[:90])

# ===== LABEL 0: BLOCK (OUT_OF_DOMAIN) - 60개 =====
block_questions = []

out_of_domain_templates = [
    "배고픈데 뭐 먹을까요?",
    "오늘 날씨 어때?",
    "주말에 뭐 할까?",
    "영화 추천해줘",
    "좋은 책 추천해줘",
    "코딩 배우는 방법 알려줘",
    "운동 방법 추천",
    "다이어트 방법",
    "여행지 추천",
    "공무원 시험 정보",
    "주식 투자 방법",
    "요리 레시피 알려줘",
    "음악 추천",
    "게임 추천",
    "쇼핑몰 추천",
    "병원 예약 방법",
    "버스 시간표",
    "지하철 노선도",
    "택시 호출 방법",
    "은행 계좌 개설",
    "신용카드 추천",
    "보험 가입 방법",
    "부동산 정보",
    "인테리어 업체 추천",
    "학원 추천",
    "취업 정보",
    "면접 준비 방법",
    "이력서 작성법",
    "연애 상담",
    "심리 상담",
]

for i in range(60):
    question = random.choice(out_of_domain_templates)
    block_questions.append({
        "input": {"question": question, "intent": "OUT_OF_DOMAIN"},
        "output": {"action": "BLOCK", "reason": "서비스 범위 밖의 질문으로 차단", "confidence": 0.95},
        "label": 0
    })

training_data.extend(block_questions[:60])

# 데이터 섞기
random.shuffle(training_data)

# 기존 파일 읽기
existing_data = []
if Path("artfacts-training/koelectra_training_dataset.sft.jsonl").exists():
    existing_data = load_jsonl(Path("artfacts-training/koelectra_training_dataset.sft.jsonl"))

# 새 데이터 추가
all_data = existing_data + training_data

# 파일 저장
output_path = Path("artfacts-training/koelectra_training_dataset.sft.jsonl")
with open(output_path, 'w', encoding='utf-8') as f:
    for item in all_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"[SUCCESS] 총 {len(training_data)}개의 학습 데이터가 생성되어 추가되었습니다.")
print(f"   - RULE_BASED (label 2): {len(rule_based_questions[:150])}개")
print(f"   - POLICY_BASED (label 1): {len(policy_based_questions[:90])}개")
print(f"   - BLOCK (label 0): {len(block_questions[:60])}개")
print(f"[FILE] 파일: {output_path}")
print(f"[TOTAL] 전체 데이터 수: {len(all_data)}개")

