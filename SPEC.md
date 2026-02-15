# Local BitNet Analyzer — Spec (doomed-based)

> 목표: **완전 로컬**, **완전 무료**, **포용적/친절**, **자연어 중심** 데이터 분석 프로그램  
> 기존 저장소는 그대로 두고, **새 저장소**를 이 스펙으로 새로 시작한다.  
> 기존 코드 재사용 없음.

---

## 0) 핵심 컨셉

- 공용 로컬 LLM 엔진: **bitnetd**
  - 내 PC에서만 도는 **localhost HTTP 서버**
  - `http://127.0.0.1:11435` 고정
  - FastAPI + SSE 기반
  - 여러 앱(분석/채팅/에이전트)이 동일 엔진에 연결 가능

- 분석 앱: **Analyzer**
  - pywebview 기반 앱 창 UI (Perplexity 스타일)
  - 단일 입력(Enter) + 파일 첨부 + 대화 로그 + 세션(새 채팅)
  - bitnetd 자동 실행/연결
  - 연결 상태는 **“연결 성공”**만 표시 (상세는 숨김)

---

## 1) 완전 무료 정책

- 외부 유료 API/서비스 사용 없음.
- 모델/엔진 바이너리를 저장소에 포함하지 않음.
- 1회 실행 시 필요한 엔진/모델은 **다운로드(반자동) + 수동 가이드 제공**.
- GitHub/HF 배포를 고려하여 weights는 레포에 포함하지 않음.

---

## 2) 설치/실행 UX 정책

### 2.1 반자동(기본)
- 앱 실행 → bitnetd 실행 시도 → `/health` 확인
- 사용자는 화면에서:
  - ✅ **연결 성공**
  - ❌ 연결 실패 시 **딱 2개 버튼만 제공**
    - `Install/Build Engine`
    - `Download Model`

> 설치/다운로드 진행률 표시 + 실패 시 재시도/리셋 안내  
> /health 상세 결과는 사용자에게 노출하지 않음(로그로만).

### 2.2 수동(항상 제공, 매우 친절)
- “Manual Install / Recovery” 가이드 제공:
  - BitNet 홈 폴더 열기 방법
  - 어떤 파일을 어디에 두는지
  - 꼬였을 때 어떤 폴더를 지우고(reset) 다시 받는지

---

## 3) 공용 BitNet 홈(공유 자산) 경로/구조

### 3.1 BitNet 홈
- `%LOCALAPPDATA%\BitNet\`

### 3.2 폴더 구조(고정)
- `bin/` : `bitnetd.exe`
- `models/` : HF snapshot 전체
- `config/` : `token.txt`, `bitnetd.json`, `manifest.json`
- `cache/` : 다운로드 임시/재개
- `logs/` : bitnetd 로그

### 3.3 Analyzer 앱 데이터(세션/로그/결과)
- `%LOCALAPPDATA%\AnalyzerApp\`
  - `history.jsonl`
  - `results/exports/`
  - `results/charts/`
  - `logs/`

> Analyzer는 설치형/포터블 둘 다 지원. 데이터는 항상 LocalAppData에 저장.

---

## 4) bitnetd 프로세스/연결 정책

- 바인딩: **127.0.0.1만** (외부 접근 불가)
- 포트: **11435 고정**
- 다중 앱 연결 지원
- 종료 정책:
  - 클라이언트 1개 이상 → 유지
  - 0개 → **즉시 종료(유예 0초)**
  - 단, 진행 중 generate가 있으면 **마무리하고 종료**
- 클라이언트 관리:
  - register + heartbeat + unregister
  - heartbeat 5초
  - TTL 15초(heartbeat 없으면 만료 처리)
- 보안 토큰:
  - `%LOCALAPPDATA%\BitNet\config\token.txt`
  - 요청 헤더 `X-Local-Token` 사용
  - `/health`는 토큰 없이 허용(상태 확인용)
  - `/generate`, `/clients/*`는 토큰 필요

---

## 5) bitnetd 기술 스택/패키징

- FastAPI + SSE
- Transformers 백엔드 (GPU 우선, 실패 시 CPU fallback)
- 패키징:
  - Analyzer: PyInstaller로 exe 패키징
  - bitnetd: PyInstaller로 로컬에서 exe 빌드/패키징
  - (정책) 레포에 엔진 바이너리를 올리지 않음 → 설치/빌드는 런타임에서 진행

---

## 6) bitnetd API 계약(확정)

### 6.1 /health
- `GET /health`
- 반환(개념):
  - status: ready / starting / not_ready / shutting_down / error
  - reasons: engine_missing / weights_missing / token_required / model_load_failed ...
  - client_count
  - (meta) version, model, uptime, token_enabled 등

> 사용자 UI에는 “연결 성공”만 표시.  
> 상세는 내부 로깅/수동 복구 가이드에만.

---

### 6.2 /clients/*
- `POST /clients/register`
  - payload: `{ client_id: uuid, app_name: "analyzer|chat|agent" }`
- `POST /clients/heartbeat`
  - payload: `{ client_id }`
- `POST /clients/unregister`
  - payload: `{ client_id }`

응답은 단순 OK/ERR + `client_count` 포함(필수).

---

### 6.3 /generate (SSE + meta 포함)
- `POST /generate`

#### Request JSON (확정)
- `prompt` (string, required)
- `stream` (bool, default true)
- options (optional, override):
  - `max_tokens` (int)
  - `temperature` (float)
  - `top_p` (float)
  - `seed` (int|null)  // null이면 랜덤
  - `repeat_penalty` (float)
  - `stop` (list[str]) // 빈 배열이면 미사용
  - `timeout_ms` (int)

기본값:
- bitnetd 내부 `analysis_default`(보수적) 프리셋 사용
- 요청이 값 지정하면 해당 필드만 override

#### Response: stream=true → SSE
이벤트 4종(확정):
- `meta` : 1회
- `delta` : 여러 회(텍스트만)
- `done` : 1회(최종 텍스트 포함 + 메타)
- `error` : 문제 시

메타 필드(확정):
- `model`
- `elapsed_ms`
- `tokens_out`
- `stop_reason`: `stop | length | timeout | error`
- `params_applied` (실제 적용 옵션)

#### Response: stream=false → JSON
- `{ text: "...", meta: {...} }`

---

## 7) Analyzer UI/UX(확정)

### 7.1 화면 구성
- 상단:
  - 새 채팅(+) / 세션 목록 토글 / 세션 삭제
  - 현재 활성 파일(Active dataset) 표시(간단)
- 본문:
  - 대화 로그(카드 형태)
- 하단:
  - 📎 다중 파일 첨부
  - 입력창(Enter 실행, Shift+Enter 줄바꿈)

### 7.2 세션 정책
- 세션 목록 + 새 채팅 + 삭제
- 삭제 전까지 유지
- `history.jsonl`로 저장(앱 재실행해도 유지)
- 제목:
  - 기본 `새 채팅 n`
  - 첫 질문 후 제목 자동 갱신(첫 문장 일부)

### 7.3 파일 첨부 정책(자동 로드)
- CSV/XLSX 중심 (PDF/PPT는 다음 마일스톤)
- 첨부된 파일은 모두 Dataset Registry에 등록
- 기본 대상: “현재 파일(Active)”
- 자연어로 targets 확장 가능(“이것도 같이 봐줘”, “첨부한 3개 비교”)

---

## 8) 자동 파일 로드(포용적/친절, 최대한 살리기)

### 8.1 CSV 인코딩 자동
- `utf-8-sig → utf-8 → cp949/euc-kr` 자동 시도
- 실패 시에도 가능한 복구 시도 후 최종 실패만 안내

### 8.2 공공데이터 잡행/헤더 자동 탐지
- 상단 제목/연도/설명 줄 존재 가능 → 헤더 후보 자동 탐지 + skiprows 자동
- 컬럼명 정리:
  - 공백/개행 제거, 중복 컬럼 suffix

### 8.3 XLSX 자동 처리
- 데이터가 가장 많은 시트 자동 선택
- 수식은 캐시된 값 우선
- 캐시 없으면 친절 안내(엑셀에서 열고 저장 후 재시도)

> 실패를 사용자 탓으로 돌리지 않음. 최대한 자동으로 살리고, 마지막에만 수동가이드.

---

## 9) 자연어 타겟/컬럼 매칭(강화 + 포용적)

### 9.1 컬럼 100~200개 대응
- 전체 컬럼 덤프 금지
- 정규화 + fuzzy + 동의어/약어 사전 확장
- 애매하면 채팅으로 좁히기

### 9.2 좁히기 UX(완화 정책)
- 1차: 후보 3개 제시 → 사용자 채팅 선택(번호/이름)
- “아니다”면 2차:
  1) 키워드 더 정확히 요청
  2) 관련 컬럼 10개 제시
  3) 유사 컬럼 전부 대상으로 할지 물음
- 최종: 새 질문 유도 + 예시 3개

### 9.3 동의어/약어 사전
- time/timestamp/datetime/date/측정일시/수집일시/발생일시...
- lat/lon/gps/위도/경도...
- temp/온도, hum/습도, voltage/전압, current/전류, power/전력, rpm, speed, accel, vib...
- id/uuid/device_id/sensor_id/robot_id...

---

## 10) Intent(자연어 기능) 1차 구성(확정)

### 10.1 지원 intent (11개)
- `summary`
- `schema`
- `filter`
- `aggregate`
- `compare`
- `plot`
- `export`
- `help`
- `columns`
- `preview`
- `validate`

### 10.2 멀티 액션(너그럽게)
- 한 문장에 여러 요청이 있으면 가능한 것부터 순차 실행
- 우선순위 예:
  - compare → validate → filter → aggregate → plot → export
- 애매한 부분만 채팅으로 좁히고, 나머지는 먼저 실행

---

## 11) 결과 카드/메타 표시(확정)

- 카드 타입 3종:
  - Text / Table(HTML) / Chart(PNG)
- 모든 카드에 `#번호` 부여(저장/참조)
- 메타는 카드 아래 작게 표시:
  - model, elapsed_ms, tokens_out, stop_reason, params_applied

---

## 12) 시각화 강화(1차에서 “분석 느낌” 확정)

- plot 요청이 애매해도 자동으로 가능하면 2~4개 차트 패키지 생성
- 기본 Top N: **20**
  - 범주 막대 Top20
  - 결측률 Top20
- 시계열 가능 시 time_column 자동 적용
- 저장: PNG 생성 후 차트 카드로 표시, 클릭 확대

---

## 13) Export(저장) 정책(확정)

- 저장은 **사용자가 요청할 때만**
- 결과는 파일 카드로 반환
- 자연어로 포맷 지정 가능: xlsx/csv/png/md/json
- 기본 범위: “이번 쿼리(직전 턴)”  
  - “시각화만”, “세션 전체”도 자연어로 가능
- Excel(xlsx) 묶음 규칙(확정):
  - INDEX / SUMMARY / TABLE_01.. / CHARTS / META
- 세션 전체 저장 기본: md+json, xlsx는 요청 시

---

## 14) 친절/포용 원칙(최상위 원칙)

- 실패를 사용자 탓으로 돌리지 않는다.
- 가능한 것은 먼저 실행하고, 애매한 것만 질문한다.
- “연결 성공” 같은 간단한 성공 피드백을 최우선.
- 고급 정보는 숨기되, 수동 복구 가이드는 매우 친절하게 제공.
- Reset(Engine/Model/All) 루트를 공식 해결책으로 제공.

---

## 15) 레포 구조(이름만 확정)

- `bitnetd/` : FastAPI+SSE 서버
- `analyzer/` : pywebview 앱
- `shared/` : 공통 타입/프로토콜(Card/Meta/API schema)
- `docs/` : 사용자 가이드(설치/수동/복구/FAQ)
