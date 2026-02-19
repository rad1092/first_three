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

