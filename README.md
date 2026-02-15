## Frontend Refactor Plan (버튼 없는 Perplexity 스타일)

### 목표
- 버튼 기반 흐름 제거
- 단일 입력(자연어) + 파일 첨부 + 대화 로그 중심 UI로 전면 개편
- 결과는 "받아보기"가 목적(사용자가 이후 조합/해석)
- 앱 창(데스크톱) 형태로 제공 (브라우저 탭 X)

### UX 규칙
1) 단일 입력창
- Enter = 실행
- Shift+Enter = 줄바꿈
- 전송 버튼(분석/실행 등) 제거

2) 파일 첨부(여러 개)
- 여러 파일 첨부 가능 (csv/xlsx 우선)
- 기본 대상 = "현재 붙인 파일(현재 활성 dataset)"
- 대화 중 새 파일이 추가되면, 자연어로 "이것도 같이 봐줘/비교해줘"로 targets 확장 가능

3) 세션(새 채팅) + 로그 유지
- 세션 목록(채팅방) 존재
- 새 채팅 = 새 세션 생성
- 이전 세션은 삭제 전까지 유지
- 사용자가 원할 때 세션 삭제 가능 (Perplexity처럼)

4) 결과 표시 규칙
- 결과는 3타입으로만 출력:
  - TextResult: 요약/근거/설명
  - TableResult: HTML table로 표시(1차)
  - ChartResult: PNG 생성 후 썸네일 표시 + 클릭 확대
- 표가 커지면 안전장치(row limit) 적용 가능

5) 파싱 실패 처리(최후 안전망)
- 자연어 처리 성능을 최우선으로 강화
- 그래도 실패 시: "내가 이해한 의도 + 부족한 정보 + 예시 3개"를 TextResult로 반환

### 설계 원칙(연결 분리)
- Router: 자연어 → intent + args + targets
- Core: intent(args, targets) → Result(Text/Table/Chart)
- UI: Result를 카드 형태로 렌더링 + 세션/첨부파일 목록 관리



