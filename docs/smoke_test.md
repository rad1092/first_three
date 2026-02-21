# Analyzer Diagnostics 스모크 테스트 가이드

## 실행 순서
1. `bitnetd`를 실행하고 `http://127.0.0.1:11435/health`가 `status=ready`인지 확인합니다.
2. Analyzer를 실행하고 상단 **진단** 버튼을 눌러 Diagnostics 패널을 엽니다.
3. `health 새로고침`으로 현재 `status/engine/model/reasons`를 확인합니다.
4. `스모크 테스트 실행` 버튼으로 아래 3개 테스트를 원클릭 실행합니다.
   - 짧은 질문
   - 표/차트 요청
   - 긴 문장 요약
5. 각 결과 카드에서 `PASS/FAIL`, `elapsed_ms`를 확인합니다.
6. FAIL 카드에서는 `오류 자세히 보기`를 펼쳐 `error/detail` 원문을 확인하고, `Copy detail`로 전체 detail을 복사합니다.

## 실패 시 체크 포인트
- `/health`가 ready가 아니면 model/exe/script path 설정과 `reasons`를 먼저 수정합니다.
- `422`가 나오면 요청 바디 형식 문제입니다(필수 필드/타입 확인).
- `503` + `0xC0000409`/`0xC0000005`가 보이면 엔진 크래시 가능성이 높습니다.
  - 안전모드 재시도(threads=1, ctx<=2048, extra_args 비움) 결과를 detail의 1차/2차 시도 블록에서 확인합니다.
- detail은 절대 요약본만 보지 말고, Diagnostics에서 원문 전체를 복사해 공유합니다.
