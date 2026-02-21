# bitnetcpp 엔진 설정 가이드

`bitnetd`에서 `bitnetcpp` 엔진을 사용하려면 `%LOCALAPPDATA%\BitNet\config\bitnetd.json`에 아래처럼 설정합니다.

## 예시 설정

```json
{
  "engine": "bitnetcpp",
  "bitnetcpp": {
    "mode": "script",
    "python_exe": "C:\\Python312\\python.exe",
    "script_path": "C:\\BitNet\\run_inference.py",
    "exe_path": "C:\\BitNet\\bin\\bitnet_cpp.exe",
    "model_path": "C:\\BitNet\\models\\ggml-model-i2_s.gguf",
    "ctx_size": 2048,
    "threads": 8,
    "extra_args": []
  }
}
```

- `engine`: `torch` 또는 `bitnetcpp`
- `mode`: `script` 또는 `exe`
- `python_exe`: `script` 모드에서 선택값 (없으면 현재 Python 실행 파일 사용)
- `script_path`: `script` 모드 필수
- `exe_path`: `exe` 모드 필수
- `model_path`: 필수
- `ctx_size`: 기본 2048
- `threads`: 기본 `os.cpu_count()/2` 수준
- `extra_args`: 엔진별 추가 플래그 전달

## 실행 인자 매핑

### script 모드

```bash
<python_exe> <script_path> -m <model_path> -p <prompt> -n <max_tokens> --temp <temperature> -c <ctx_size> -t <threads>
```

### exe 모드

```bash
<exe_path> -m <model_path> -p <prompt> -n <max_tokens> --temp <temperature> -c <ctx_size> -t <threads>
```

추가 옵션은 `extra_args`로 뒤에 붙습니다.

## 동작/제약

- `stream=false`: 서브프로세스 실행 결과를 정제해 최종 텍스트를 반환합니다.
- `stream=true`: 내부적으로 전체 결과를 만든 뒤 고정 길이로 나눠 `delta` 이벤트를 전송하는 pseudo-stream 방식입니다.
- `top_p`, `repeat_penalty`는 bitnet.cpp CLI가 지원하지 않을 수 있으며, 미지원 시 무시될 수 있습니다.
- `stop`은 CLI 미지원일 경우 bitnetd에서 후처리로 잘라 `stop_reason=stop`을 반영합니다.

## 준비 상태(health reasons)

경로/설정이 잘못되면 `/health` reasons에 아래 값이 나타납니다.

- `bitnetcpp_model_missing`
- `bitnetcpp_script_missing`
- `bitnetcpp_executable_missing`
- `bitnetcpp_mode_invalid`

## Windows 빌드/설치 주의 (allowlist 환경)

Windows에서 bitnet.cpp 빌드 시 Visual Studio 설치/다운로드가 `aka.ms` 등 추가 도메인으로 연결될 수 있어,
allowlist 제한 환경에서는 설치가 차단될 수 있습니다.

권장 대응:
1. 오프라인 레이아웃 설치 사용
2. 네트워크 allowlist 확장
3. 이미 설치된 C++ 툴체인 재사용
