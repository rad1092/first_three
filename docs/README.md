# 문서 인덱스

- [수동 설치 가이드](./MANUAL_INSTALL.md)
- [리셋/문제해결](./RESET_TROUBLESHOOT.md)

Phase 0 문서는 목차 중심 초안이며, Phase 1에서 절차를 구체화합니다.

## Phase 8.1 검증

bitnetd 시작 후 `/health`가 `ready`이면 BitNet 모델 다운로드/manifest 기록/로드가 완료된 상태입니다.
초기 로드 이전에는 `starting`과 `model_not_loaded` 이유가 보일 수 있습니다.
모델 다운로드나 로드 실패 시 `/health`는 `error`와 `model_load_failed` 이유를 반환해야 합니다.
모델 파일은 레포가 아니라 `%LOCALAPPDATA%\BitNet\models\` 하위에만 생성되어야 합니다.
BitNet 양자화 모델 로딩 시 환경에 따라 `accelerate`가 필요할 수 있습니다.
`accelerate`가 없으면 startup 로드가 실패하고 `/health`는 `error/model_load_failed`로 남을 수 있습니다.
의존성 설치 후 동일한 실행 흐름에서 로드가 완료되면 `/health`는 `ready`로 전환됩니다.

Windows에서 "cl is not found" 오류가 보이면 Visual Studio Build Tools(C++ Build Tools, cl.exe)가 필요할 수 있습니다.
bitnetd는 호환을 위해 torch compile/inductor 비활성 safe mode를 자동 적용합니다.
safe mode에서는 추론 성능이 다소 느려질 수 있습니다.

## Phase 8.2 스모크 테스트

`/generate`는 `prompt` 기반 요청으로 `stream=false`이면 `{text, meta}` JSON, `stream=true`이면 SSE(`meta -> delta... -> done`)를 반환해야 합니다.
SSE의 `delta` 이벤트 payload는 반드시 `{"delta":"..."}` 단일 필드여야 하며, 최종 `done.text`와 누적 delta 텍스트가 같아야 합니다.
`timeout_ms` 또는 `stop` 조건 발생 시 `meta.stop_reason`이 각각 `timeout`/`stop`으로 구분되어야 합니다.

## Swagger 토큰 입력

`/docs`에서 **Authorize**를 누른 뒤 `%LOCALAPPDATA%\BitNet\config\token.txt` 값을 `X-Local-Token`으로 입력하면 됩니다.
한 번 인증하면 `/generate`와 `/clients/*` 요청에 헤더가 자동 적용되어 테스트할 수 있습니다.
`/health`는 기존처럼 토큰 없이 호출 가능합니다.

## Phase 8.3 안내

Analyzer는 Active 데이터셋이 없어도 bitnetd 연결이 살아 있으면 자연어 대화를 이어갈 수 있습니다.
요약/그래프/검증 같은 분석 요청은 데이터셋이 없을 때 파일 첨부가 필요한 이유와 다음 행동을 한국어로 안내합니다.
bitnetd 연결 또는 토큰 문제가 있으면 짧은 연결 안내 문구를 보여주고, 데이터셋이 있는 분석 흐름은 기존대로 유지됩니다.

## Phase 8.3 검증

Active가 없는 상태에서 "안녕"을 보내면 한국어 2~4문장 대화 답변이 나오고 영어/코드블록이 없어야 합니다.
Active가 없는 상태에서 "요약해줘"를 보내면 파일 첨부 필요 안내와 1)~3) 다음 행동이 보여야 합니다.
같은 실패 문구가 연속으로 누적되면 중복 방지 로직 미동작으로 간주합니다.
