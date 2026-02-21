from __future__ import annotations

import os
import shlex
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from ..config import BitnetCppConfig
from .base import EngineError, GenerateParams


@dataclass(slots=True)
class _ProcessAttempt:
    name: str
    cmd: list[str]
    cwd: str | None
    prompt_mode: str
    prompt_file_flag: str | None
    prompt_file_created: bool
    completed: subprocess.CompletedProcess[str] | None = None
    os_error: str | None = None


class BitnetCppEngine:
    engine_id = "bitnetcpp"
    stream_supported = True

    def __init__(self, config: BitnetCppConfig) -> None:
        self._config = config
        self._prompt_file_flag: str | None = None
        self._prompt_file_checked: bool = False

    def model_label(self) -> str:
        model_path = Path(self._config.model_path) if self._config.model_path else None
        if model_path and model_path.name:
            return f"bitnetcpp:{model_path.name}"
        return "bitnetcpp:unknown"

    def is_ready(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        mode = self._config.mode
        if not self._config.model_path or not Path(self._config.model_path).exists():
            reasons.append("bitnetcpp_model_missing")

        if mode == "script":
            if not self._config.script_path or not Path(self._config.script_path).exists():
                reasons.append("bitnetcpp_script_missing")
        elif mode == "exe":
            if not self._config.exe_path or not Path(self._config.exe_path).exists():
                reasons.append("bitnetcpp_executable_missing")
        else:
            reasons.append("bitnetcpp_mode_invalid")

        return len(reasons) == 0, reasons

    def ensure_loaded(self, snapshot_path: str) -> None:
        return

    @staticmethod
    def _tail_lines(text: str, *, max_lines: int = 60, max_chars: int = 3200) -> str:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        tail = "\n".join(lines[-max_lines:]).strip()
        if len(tail) > max_chars:
            tail = tail[-max_chars:]
        return tail

    def _resolve_cwd(self) -> str | None:
        cfg = self._config
        if cfg.mode == "exe" and cfg.exe_path:
            return str(Path(cfg.exe_path).parent)
        if cfg.mode == "script" and cfg.script_path:
            return str(Path(cfg.script_path).parent)
        return None

    @staticmethod
    def _is_crash_code(exit_code: int) -> bool:
        return (exit_code & 0xFFFFFFFF) in {0xC0000409, 0xC0000005, 0xC000001D}

    @staticmethod
    def _format_exit_code(exit_code: int) -> tuple[str, str | None]:
        unsigned_code = exit_code & 0xFFFFFFFF
        display = f"exit_code={exit_code} (0x{unsigned_code:08X})"
        crash_hint = None
        if BitnetCppEngine._is_crash_code(exit_code):
            crash_hint = "Windows 프로세스 크래시 가능성: 런타임/스택버퍼/접근위반 등 치명적 오류를 확인하세요."
        return display, crash_hint

    @staticmethod
    def _is_likely_log_line(line: str) -> bool:
        normalized = line.strip().lower()
        if not normalized:
            return False
        noisy_prefixes = (
            "build:",
            "main:",
            "llm_",
            "llama_",
            "system_info:",
            "warning:",
            "load_",
            "print_info:",
            "common_",
            "ggml_",
            "sampling:",
            "prompt eval",
            "eval time",
            "total time",
        )
        return normalized.startswith(noisy_prefixes)

    def _help_command(self) -> list[str]:
        cfg = self._config
        if cfg.mode == "script":
            return [cfg.python_exe, cfg.script_path, "--help"]
        if cfg.mode == "exe":
            return [cfg.exe_path, "--help"]
        return []

    def _detect_prompt_file_flag(self) -> str | None:
        if self._prompt_file_checked:
            return self._prompt_file_flag

        detected: str | None = None
        help_cmd = self._help_command()
        if help_cmd:
            try:
                completed = subprocess.run(
                    help_cmd,
                    cwd=self._resolve_cwd(),
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                    encoding="utf-8",
                    errors="replace",
                    env=os.environ.copy(),
                )
                help_text = f"{completed.stdout}\n{completed.stderr}".lower()
                if "-f," in help_text or " -f " in help_text or "\n-f" in help_text:
                    detected = "-f"
                elif "--file" in help_text:
                    detected = "--file"
            except Exception:  # noqa: BLE001
                detected = None

        self._prompt_file_flag = detected
        self._prompt_file_checked = True
        return self._prompt_file_flag

    def _build_failure_detail(
        self,
        *,
        cmd: list[str],
        exit_code: int,
        stdout: str,
        stderr: str,
        reason: str,
        prompt_mode: str,
        prompt_file_flag: str | None,
        prompt_file_created: bool,
    ) -> str:
        exit_summary, crash_hint = self._format_exit_code(exit_code)
        preview = " ".join(shlex.quote(v) for v in cmd)
        parts = [
            reason,
            exit_summary,
            f"command={preview}",
            f"prompt_mode={prompt_mode}",
            f"prompt_file_flag={prompt_file_flag or 'none'}",
            f"prompt_file_created={'true' if prompt_file_created else 'false'}",
        ]
        if crash_hint:
            parts.append(crash_hint)
        stdout_tail = self._tail_lines(stdout)
        stderr_tail = self._tail_lines(stderr)
        if stderr_tail:
            parts.append(f"stderr_tail:\n{stderr_tail}")
        if stdout_tail:
            parts.append(f"stdout_tail:\n{stdout_tail}")
        return "\n\n".join(parts)

    def _build_command(
        self,
        *,
        prompt_arg: list[str],
        params: GenerateParams,
        threads_override: int | None = None,
        ctx_size_override: int | None = None,
        extra_args_override: list[str] | None = None,
    ) -> list[str]:
        cfg = self._config
        threads = threads_override if threads_override is not None else int(cfg.threads)
        ctx_size = ctx_size_override if ctx_size_override is not None else int(cfg.ctx_size)
        selected_extra = cfg.extra_args if extra_args_override is None else extra_args_override
        common_args = [
            "-m",
            cfg.model_path,
            *prompt_arg,
            "-n",
            str(params.max_tokens),
            "--temp",
            str(params.temperature),
            "-c",
            str(ctx_size),
            "-t",
            str(threads),
        ]
        extra = [str(v) for v in selected_extra]

        if cfg.mode == "script":
            return [cfg.python_exe, cfg.script_path, *common_args, *extra]
        if cfg.mode == "exe":
            return [cfg.exe_path, *common_args, *extra]
        raise EngineError(
            error="bitnetcpp mode 설정이 잘못되었습니다.",
            detail="bitnetcpp_mode_invalid",
            status_code=503,
        )

    def _run_attempt(self, attempt: _ProcessAttempt, timeout_ms: int | None) -> _ProcessAttempt:
        try:
            attempt.completed = subprocess.run(
                attempt.cmd,
                cwd=attempt.cwd,
                capture_output=True,
                text=True,
                timeout=max(1, int((timeout_ms or 60000) / 1000)),
                check=False,
                encoding="utf-8",
                errors="replace",
                env=os.environ.copy(),
            )
            return attempt
        except subprocess.TimeoutExpired:
            raise EngineError(
                error="bitnetcpp 실행 시간이 초과되었습니다.",
                detail=f"attempt={attempt.name}",
                status_code=503,
            )
        except OSError as exc:
            attempt.os_error = str(exc)
            return attempt

    def _attempt_detail(self, attempt: _ProcessAttempt) -> str:
        if attempt.os_error:
            preview = " ".join(shlex.quote(v) for v in attempt.cmd)
            return "\n\n".join(
                [
                    f"attempt={attempt.name}",
                    f"os_error={attempt.os_error}",
                    f"command={preview}",
                    f"prompt_mode={attempt.prompt_mode}",
                    f"prompt_file_flag={attempt.prompt_file_flag or 'none'}",
                    f"prompt_file_created={'true' if attempt.prompt_file_created else 'false'}",
                ]
            )

        completed = attempt.completed
        assert completed is not None
        return self._build_failure_detail(
            cmd=attempt.cmd,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            reason=f"attempt={attempt.name}",
            prompt_mode=attempt.prompt_mode,
            prompt_file_flag=attempt.prompt_file_flag,
            prompt_file_created=attempt.prompt_file_created,
        )

    def _clean_output(self, stdout: str, stderr: str) -> str:
        out_lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        err_lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        joined_err = "\n".join(err_lines).lower()
        joined_out = "\n".join(out_lines).lower()

        if "usage:" in joined_out or "usage:" in joined_err:
            raise EngineError(
                error="bitnetcpp 실행 인자가 올바르지 않습니다.",
                detail=(err_lines[0] if err_lines else "usage_detected"),
                status_code=503,
            )

        if not out_lines and not err_lines:
            raise EngineError(
                error="bitnetcpp 출력이 비어 있습니다.",
                detail="empty_stdout_stderr",
                status_code=503,
            )

        filtered_out_lines = [line for line in out_lines if not self._is_likely_log_line(line)]
        filtered_err_lines = [line for line in err_lines if not self._is_likely_log_line(line)]

        focus_lines = filtered_out_lines[-60:] if filtered_out_lines else out_lines[-60:]
        if not focus_lines and filtered_err_lines:
            focus_lines = filtered_err_lines[-60:]
        candidate = "\n".join(focus_lines).strip()
        if len(candidate) < 24:
            candidate = "\n".join(filtered_out_lines or out_lines).strip()
        if len(candidate) < 24 and filtered_err_lines:
            candidate = "\n".join(filtered_err_lines).strip()
        if not candidate:
            stdout_tail = self._tail_lines(stdout)
            stderr_tail = self._tail_lines(stderr)
            detail_parts = ["empty_clean_output"]
            if stderr_tail:
                detail_parts.append(f"stderr_tail:\n{stderr_tail}")
            if stdout_tail:
                detail_parts.append(f"stdout_tail:\n{stdout_tail}")
            raise EngineError(
                error="bitnetcpp 출력 정제 결과가 비어 있습니다.",
                detail="\n\n".join(detail_parts),
                status_code=503,
            )
        return candidate

    def _apply_stop(self, text: str, stop_markers: list[str]) -> tuple[str, str]:
        if not stop_markers:
            return text, "length"
        cut_pos: int | None = None
        for marker in stop_markers:
            if not marker:
                continue
            idx = text.find(marker)
            if idx >= 0 and (cut_pos is None or idx < cut_pos):
                cut_pos = idx
        if cut_pos is None:
            return text, "length"
        return text[:cut_pos], "stop"

    def generate_nonstream(self, prompt: str, *, params: GenerateParams) -> tuple[str, str, int]:
        ok, reasons = self.is_ready()
        if not ok:
            raise EngineError(
                error="bitnetcpp 엔진 준비 상태가 아닙니다.",
                detail=",".join(reasons),
                status_code=503,
            )

        cwd = self._resolve_cwd()
        prompt_file_needed = (not prompt.isascii()) or len(prompt) >= 800
        detected_flag = self._detect_prompt_file_flag()
        prompt_file_flag = detected_flag if detected_flag else "-f"
        prompt_mode = "inline"
        prompt_file_created = False
        tmp_path: Path | None = None
        prompt_arg: list[str] = ["-p", prompt]

        try:
            if prompt_file_needed:
                prompt_mode = "file"
                base_dir = Path(cwd) if cwd else Path.cwd()
                tmp_path = base_dir / f"prompt_tmp_{uuid4().hex}.txt"
                tmp_path.write_text(prompt, encoding="utf-8", newline="\n")
                prompt_file_created = True
                prompt_arg = [prompt_file_flag, str(tmp_path)]

            first_attempt = self._run_attempt(
                _ProcessAttempt(
                    name="1차 시도",
                    cmd=self._build_command(prompt_arg=prompt_arg, params=params),
                    cwd=cwd,
                    prompt_mode=prompt_mode,
                    prompt_file_flag=prompt_file_flag if prompt_mode == "file" else None,
                    prompt_file_created=prompt_file_created,
                ),
                params.timeout_ms,
            )
            if first_attempt.os_error:
                raise EngineError(
                    error="bitnetcpp 서브프로세스를 시작하지 못했습니다.",
                    detail=self._attempt_detail(first_attempt),
                    status_code=503,
                )

            completed = first_attempt.completed
            assert completed is not None

            if completed.returncode != 0 and self._is_crash_code(completed.returncode):
                second_attempt = self._run_attempt(
                    _ProcessAttempt(
                        name="2차 시도(안전모드)",
                        cmd=self._build_command(
                            prompt_arg=prompt_arg,
                            params=params,
                            threads_override=1,
                            ctx_size_override=min(int(self._config.ctx_size), 2048),
                            extra_args_override=[],
                        ),
                        cwd=cwd,
                        prompt_mode=prompt_mode,
                        prompt_file_flag=prompt_file_flag if prompt_mode == "file" else None,
                        prompt_file_created=prompt_file_created,
                    ),
                    params.timeout_ms,
                )
                if second_attempt.os_error:
                    raise EngineError(
                        error="bitnetcpp 실행에 실패했습니다.",
                        detail="\n\n".join(
                            [
                                "bitnetcpp 프로세스 크래시 후 안전모드 재시도 시작 실패",
                                self._attempt_detail(first_attempt),
                                self._attempt_detail(second_attempt),
                            ]
                        ),
                        status_code=503,
                    )

                completed = second_attempt.completed
                assert completed is not None
                if completed.returncode != 0:
                    raise EngineError(
                        error="bitnetcpp 실행에 실패했습니다.",
                        detail="\n\n".join(
                            [
                                "bitnetcpp 프로세스 크래시 후 안전모드 재시도까지 실패",
                                self._attempt_detail(first_attempt),
                                self._attempt_detail(second_attempt),
                            ]
                        ),
                        status_code=503,
                    )

            elif completed.returncode != 0:
                raise EngineError(
                    error="bitnetcpp 실행에 실패했습니다.",
                    detail=self._attempt_detail(first_attempt),
                    status_code=503,
                )

            text = self._clean_output(completed.stdout, completed.stderr)
            text, stop_reason = self._apply_stop(text, params.stop)
            text = text.strip()
            if not text:
                raise EngineError(
                    error="bitnetcpp가 빈 텍스트를 반환했습니다.",
                    detail="empty_text",
                    status_code=503,
                )
            return text, stop_reason, 0
        finally:
            if tmp_path is not None:
                with suppress(Exception):
                    tmp_path.unlink()

    def generate_stream(self, prompt: str, *, params: GenerateParams) -> Iterator[str]:
        text, _, _ = self.generate_nonstream(prompt, params=params)
        chunk_size = 48

        def _iter() -> Iterator[str]:
            for i in range(0, len(text), chunk_size):
                yield text[i : i + chunk_size]

        return _iter()

    def debug_command_preview(self, prompt: str, params: GenerateParams) -> str:
        prompt_arg = ["-p", prompt]
        return " ".join(shlex.quote(v) for v in self._build_command(prompt_arg=prompt_arg, params=params))
