from __future__ import annotations

from urllib.parse import urlparse

import uvicorn

from shared.constants import BITNETD_BASE_URL


def _host_port_from_base_url() -> tuple[str, int]:
    parsed = urlparse(BITNETD_BASE_URL)
    host = parsed.hostname
    port = parsed.port

    if parsed.scheme != "http" or host != "127.0.0.1" or port != 11435:
        raise RuntimeError("BITNETD_BASE_URL must be http://127.0.0.1:11435")

    return host, port


def main() -> None:
    host, port = _host_port_from_base_url()
    uvicorn.run("bitnetd.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
