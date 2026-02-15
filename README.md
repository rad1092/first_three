# BitNet Tools

BitNet/Ollama 기반의 **로컬 데이터 분석 보조 도구**입니다.  
CSV를 중심으로 Excel/문서(PDF, DOCX, PPTX) 입력을 정규화해 요약·비교·리포트·시각화·질문 기반 분석 파이프라인으로 연결합니다.

---

## 저장소 점검 결과 요약

### 1) 현재 제공 기능

- CLI 분석 명령
  - `analyze`: 단일 파일 분석 payload + (옵션) Ollama 즉시 질의
  - `report`: Markdown 보고서 생성
  - `multi-analyze`: 다중 CSV 통합 분석(JSON/MD)
  - `compare`: 전/후 CSV 분포 비교
  - `doctor`: 로컬 환경 진단
  - `ui`, `desktop`: 웹 UI/데스크톱 UI 실행
- 입력 전처리
  - CSV 직접 입력
  - Excel(`.xlsx`) base64 로드, 시트 선택, CSV 정규화
  - 문서(`.pdf`, `.docx`, `.pptx`) 표 추출 후 분석 요청 변환
- 분석 코어
  - 결측/타입/기초 수치 통계(평균/최소/최대) 생성
  - 다중 CSV 프로파일 집계 및 캐시(`.bitnet_cache`) 활용
  - 룰 기반 이상 징후 설명 후보 생성(결측 집중/편중/단위 불일치/최근 급변)
  - 지리 좌표 유효성 검사 및 의심 레코드 산출
- 시각화/추천
  - 다중 CSV 차트 생성(환경에 따라 matplotlib 기반)
  - 질문/스키마 기반 차트 타입 추천
- API/UI
  - 로컬 HTTP 서버에서 분석/전처리/비교/플래너 관련 엔드포인트 제공
  - 브라우저 UI 정적 파일 제공

### 2) 어떤 분석이 가능한가

- 데이터 품질 분석: 결측, 타입 일관성, 지배 카테고리 편향, 단위 혼재
- 기술통계 분석: 컬럼별 수치 통계, 그룹 합계/랭킹/샘플 추출
- 다중 파일 분석: 여러 CSV의 프로파일 비교/통합 리포트 생성
- 전후 비교 분석: before/after CSV 분포 차이 확인
- 지리 데이터 검증: 위경도 범위/이상치 플래깅
- 문서/엑셀 표 기반 분석: 원본을 CSV 형태로 정규화한 뒤 동일 파이프라인 적용

### 3) 자연어 처리(NLP) 수준 진단

이 프로젝트의 NLP는 **경량 규칙 기반 + LLM 연동 보조형** 수준입니다.

- 강점
  - 한/영 키워드 기반 의도 파싱(`top N`, `샘플 N`, `임계값`, `전후` 등)
  - 스키마 시맨틱 별칭 매핑(질문 용어 ↔ 실제 컬럼명)
  - 분석 계획(플랜) 자동 구성 후 실행으로 질문-분석 연결
  - 최종 설명은 Ollama 모델(BitNet 등)로 확장 가능
- 한계
  - 복잡 문장 의미 해석, 다중 의도 분해, 문맥 추론은 제한적
  - 모델 없이 동작하는 NLP는 정규식/사전 중심이라 표현 변화에 취약
  - 고급 NLU(개체/관계 추출, 추론 체인, 다중 턴 대화 메모리)는 별도 구현 없음

> 결론: “완전한 자연어 이해 엔진”보다는, **정형 데이터 분석 자동화에 최적화된 실무형 NLP 어댑터**에 가깝습니다.

---

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

CLI 도움말:

```bash
python -m bitnet_tools.cli --help
```

주요 예시:

```bash
# 단일 분석 payload
bitnet-analyze analyze sample.csv --question "핵심 지표 요약" --out payload.json

# Markdown 보고서
bitnet-analyze report sample.csv --question "핵심 요약" --out analysis_report.md

# 다중 CSV 분석
bitnet-analyze multi-analyze a.csv b.csv --question "지역별 차이" --out-json multi.json --out-report multi.md

# 전/후 비교
bitnet-analyze compare --before before.csv --after after.csv --out compare.json

# 웹 UI
bitnet-analyze ui --host 127.0.0.1 --port 8765

# 환경 진단
bitnet-analyze doctor --model bitnet:latest
```

---

## 입력 지원 범위

- `csv`: 텍스트/파일 기반 직접 분석
- `excel`: `.xlsx` 지원(시트 선택 후 CSV 정규화)
- `document`: `.pdf`, `.docx`, `.pptx`에서 표 추출 후 분석

제약:
- `.xls`(구형 바이너리 엑셀) 미지원
- 표/헤더 품질이 낮은 문서는 추출 실패 가능
- 대형 파일/복잡 문서는 전처리 단계에서 시간 증가 가능

---

## 테스트 상태

현재 테스트 스위트 기준, 핵심 기능(분석/비교/플래너/문서추출/웹/UI 계약/오프라인 번들)이 통과하는 상태입니다.

```bash
pytest -q
```

