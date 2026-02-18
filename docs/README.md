# 문서 인덱스

- [수동 설치 가이드](./MANUAL_INSTALL.md)
- [리셋/문제해결](./RESET_TROUBLESHOOT.md)

Phase 0 문서는 목차 중심 초안이며, Phase 1에서 절차를 구체화합니다.

## Phase 8-2 실행 및 스모크 테스트

> 토큰 파일: `%LOCALAPPDATA%\\BitNet\\config\\token.txt`  
> 요청 헤더: `X-Local-Token`

### PowerShell (필수)

1) `/health` (무토큰 허용)

```powershell
Invoke-RestMethod -Method GET -Uri "http://127.0.0.1:11435/health"
```

2) `/generate` stream=false (JSON)

```powershell
$token = Get-Content "$env:LOCALAPPDATA\BitNet\config\token.txt" -Raw
$headers = @{
  "Content-Type" = "application/json"
  "X-Local-Token" = $token.Trim()
}
$body = @{
  prompt = "데이터 요약을 3줄로 작성해줘."
  stream = $false
  max_tokens = 96
  temperature = 0
} | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:11435/generate" -Headers $headers -Body $body
```

3) `/generate` stream=true (SSE)

```powershell
$token = Get-Content "$env:LOCALAPPDATA\BitNet\config\token.txt" -Raw
$body = '{"prompt":"CSV 이상치 탐지 절차를 설명해줘.","stream":true,"max_tokens":96,"temperature":0.7,"top_p":0.9}'
curl.exe -N "http://127.0.0.1:11435/generate" `
  -H "Content-Type: application/json" `
  -H "X-Local-Token: $($token.Trim())" `
  -d $body
```

### CMD (필수)

```cmd
for /f "usebackq delims=" %t in (`type "%LOCALAPPDATA%\BitNet\config\token.txt"`) do curl -sS http://127.0.0.1:11435/generate -H "Content-Type: application/json" -H "X-Local-Token: %t" -d "{\"prompt\":\"요약해줘\",\"stream\":false,\"max_tokens\":64,\"temperature\":0}"
```

### Git Bash (선택)

```bash
TOKEN=$(cat "$LOCALAPPDATA/BitNet/config/token.txt")
curl -N http://127.0.0.1:11435/generate \
  -H "Content-Type: application/json" \
  -H "X-Local-Token: $TOKEN" \
  -d '{"prompt":"컬럼 품질 검사 체크리스트를 작성해줘.","stream":true,"max_tokens":64}'
```
