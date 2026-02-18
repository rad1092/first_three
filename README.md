# Local BitNet Analyzer (Phase 0)

이 저장소는 **SPEC.md / IMPLEMENTATION_PLAN.md** 계약을 기준으로 시작하는 초기 골격(Phase 0)입니다.

## 목표
- 완전 로컬 기반 분석 앱/엔진 아키텍처의 시작점 마련
- 구조 고정: `bitnetd`, `analyzer`, `shared`, `docs`
- 공통 상수/경로를 `shared`에 단일화

## 저장소 구성
- `bitnetd/` : FastAPI + SSE 서버 최소 골격
- `analyzer/` : pywebview 앱 최소 골격
- `shared/` : 공통 상수/경로/타입 확장 기반
- `docs/` : 수동 설치/복구/FAQ 초안

## 경로 규칙 (Windows)
- BitNet 홈: `%LOCALAPPDATA%\BitNet\`
  - `bin/`, `models/`, `config/`, `cache/`, `logs/`
- Analyzer 앱 데이터: `%LOCALAPPDATA%\AnalyzerApp\`
  - `history.jsonl`, `results/`, `logs/`

## 고정 주소
- bitnetd base URL: `http://127.0.0.1:11435`

## 다음 단계 (Phase 1)
- `bitnetd` FastAPI 엔드포인트(`/health`, `/clients/*`, `/generate`) 실제 구현
- `analyzer` 세션/대화/파일첨부 UI 및 bitnetd 연결 로직 구현
- shared 타입/프로토콜 스키마 확장
- docs 상세 설치/복구 절차 보강

## bitnetd `/generate` 호출 예시 (Phase 8)

토큰 파일 경로는 `bitnetd/security.py` 기준으로 `%LOCALAPPDATA%\\BitNet\\config\\token.txt` 입니다.

```bash
TOKEN=$(cat "$LOCALAPPDATA/BitNet/config/token.txt")
curl -sS http://127.0.0.1:11435/generate \
  -H "Content-Type: application/json" \
  -H "X-Local-Token: $TOKEN" \
  -d '{
    "prompt": "데이터 분석 요약을 3줄로 작성해줘.",
    "stream": false,
    "max_tokens": 128,
    "temperature": 0
  }'
```

```bash
TOKEN=$(cat "$LOCALAPPDATA/BitNet/config/token.txt")
curl -N http://127.0.0.1:11435/generate \
  -H "Content-Type: application/json" \
  -H "X-Local-Token: $TOKEN" \
  -d '{
    "prompt": "CSV 컬럼 유효성 검사 절차를 단계별로 설명해줘.",
    "stream": true,
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```
