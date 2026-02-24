# Local BitNet Analyzer

로컬 전용 분석 앱(`analyzer`) + 로컬 LLM 서버(`bitnetd`) 프로젝트입니다.

> 이 README는 "추정"이 아니라 **현재 코드/문서 검토 결과 기준**으로 작성했습니다.

## 1) 현재 상태 한눈에 보기 (코드 검토 기준)

- 기준 문서/코드
  - 요구사항: `SPEC.md`, `IMPLEMENTATION_PLAN.md`
  - 서버: `bitnetd/app.py`, `bitnetd/state.py`, `bitnetd/engines/*`
  - 앱: `analyzer/app.py`, `analyzer/router.py`, `analyzer/executor.py`, `analyzer/history.py`

### 구현 확인 체크리스트

- ✅ bitnetd localhost 고정 (`127.0.0.1:11435`) 실행 경로 존재
- ✅ `/health`, `/clients/register|heartbeat|unregister`, `/generate` 엔드포인트 구현
- ✅ `/clients/*`, `/generate` 토큰 인증(`X-Local-Token`), `/health` 무인증
- ✅ `/generate`의 stream/non-stream 응답 분기 구현(SSE + JSON)
- ✅ Analyzer 세션 히스토리(`history.jsonl`) append/load 로직 구현
- ✅ Router intent(요약/스키마/필터/집계/비교/플롯/내보내기/도움/컬럼/미리보기/검증) 분기 구현
- ✅ 실행기에서 summary/schema/preview/validate/compare/plot 등 카드 생성 로직 구현
- ⚠️ 자동 테스트(`pytest`)는 현재 0개(회귀 방지 자동화는 아직 부족)

## 2) 저장소 구성

- `bitnetd/` : FastAPI 서버, 클라이언트 생명주기, 토큰 인증, generate
- `analyzer/` : pywebview 앱(UI/라우팅/실행기/히스토리)
- `shared/` : 공통 상수/경로
- `docs/` : 설치/복구/스모크 테스트/엔진 연동 문서

## 3) 핵심 동작

### bitnetd
- 기본 주소: `http://127.0.0.1:11435`
- 엔드포인트
  - `GET /health`
  - `POST /clients/register`
  - `POST /clients/heartbeat`
  - `POST /clients/unregister`
  - `POST /generate`
- 인증 정책
  - `/health`: 토큰 없이 호출 가능
  - `/clients/*`, `/generate`: `%LOCALAPPDATA%\BitNet\config\token.txt` 값을 `X-Local-Token` 헤더로 전달
- `/generate` 응답 형식
  - `stream=false` → `{text, meta}` JSON
  - `stream=true` → SSE(`meta`, `delta`, `done`, `error`)

### analyzer
- pywebview 기반 채팅형 분석 UI
- 세션/메시지/파일 이벤트를 `history.jsonl`에 기록
- CSV/XLSX 로드 + 데이터셋 레지스트리 관리
- 자연어 라우팅 후 카드(Text/Table/Chart) 기반 결과 제공

## 4) 경로 규칙 (Windows)

- BitNet 홈: `%LOCALAPPDATA%\BitNet\`
  - `bin/`, `models/`, `config/`, `cache/`, `logs/`
- Analyzer 앱 데이터: `%LOCALAPPDATA%\AnalyzerApp\`
  - `history.jsonl`, `results/exports/`, `results/charts/`, `logs/`

## 5) 빠른 실행

```bash
pip install -r requirements.txt
python -m bitnetd
```

다른 터미널에서:

```bash
python -m analyzer
```

## 6) 문서

- 문서 인덱스: `docs/README.md`
- 수동 설치: `docs/MANUAL_INSTALL.md`
- 리셋/문제해결: `docs/RESET_TROUBLESHOOT.md`
- bitnetcpp 엔진 설정: `docs/bitnetcpp.md`
- Diagnostics 스모크 테스트: `docs/smoke_test.md`
- 레포 검토 보고서: `docs/REPO_REVIEW.md`

## 7) 로드맵 기준 문서

- `SPEC.md`
- `IMPLEMENTATION_PLAN.md`
- `IMPLEMENTATION.md`
