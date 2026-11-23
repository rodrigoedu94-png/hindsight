"""Optional Sentry error/trace reporting, armed by ``SENTRY_DSN``.

Unset or empty ``SENTRY_DSN`` is a clean no-op: ``sentry_sdk`` is not even imported, so
the package stays an optional dependency and uvicorn ``--workers N`` spawn stays cheap
(see ``hindsight_api/__init__.py`` for why top-level imports matter here).

Environment:
    SENTRY_DSN                 project DSN (never logged)
    SENTRY_ENVIRONMENT         default ``production``
    SENTRY_RELEASE             default ``hindsight-oracle@<version>``
    SENTRY_TRACES_SAMPLE_RATE  default ``0.1``, clamped to [0, 1]

Privacy: no user info, cookies, request bodies or GenAI/MCP payloads are collected, and
``before_send`` additionally strips credential headers.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

ENV_SENTRY_DSN = "SENTRY_DSN"
ENV_SENTRY_ENVIRONMENT = "SENTRY_ENVIRONMENT"
ENV_SENTRY_RELEASE = "SENTRY_RELEASE"
ENV_SENTRY_TRACES_SAMPLE_RATE = "SENTRY_TRACES_SAMPLE_RATE"

_SENSITIVE_HEADERS = frozenset({"authorization", "x-api-key", "cookie", "set-cookie"})

_installed = False


def _parse_sample_rate(raw: str | None, default: float = 0.1) -> float:
    """Parse a sample rate; fall back to ``default`` on junk, clamp to [0, 1]."""
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return min(1.0, max(0.0, value))


def _before_send(event: dict[str, Any], _hint: dict[str, Any]) -> dict[str, Any]:
    """Strip credential headers, cookies and request bodies from an outgoing event."""
    request = event.get("request")
    if isinstance(request, dict):
        headers = request.get("headers")
        if isinstance(headers, dict):
            request["headers"] = {k: v for k, v in headers.items() if str(k).lower() not in _SENSITIVE_HEADERS}
        request.pop("cookies", None)
        request.pop("data", None)
    return event


def install() -> bool:
    """Initialise Sentry once per process. Returns True when reporting is active."""
    global _installed
    if _installed:
        return True

    dsn = os.getenv(ENV_SENTRY_DSN, "").strip()
    if not dsn:
        return False

    try:
        import sentry_sdk
    except ImportError:
        logger.warning("SENTRY_DSN is set but sentry-sdk is not installed; Sentry reporting disabled")
        return False

    from hindsight_api import __version__

    environment = os.getenv(ENV_SENTRY_ENVIRONMENT, "").strip() or "production"
    release = os.getenv(ENV_SENTRY_RELEASE, "").strip() or f"hindsight-oracle@{__version__}"
    traces_sample_rate = _parse_sample_rate(os.getenv(ENV_SENTRY_TRACES_SAMPLE_RATE), default=0.1)

    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            release=release,
            traces_sample_rate=traces_sample_rate,
            data_collection={
                "user_info": False,
                "cookies": {"mode": "off"},
                "http_bodies": [],
                "gen_ai": {"inputs": False, "outputs": False},
                "mcp": {"inputs": False, "outputs": False},
                "stack_frame_variables": False,
            },
            before_send=_before_send,
        )
    except Exception:  # noqa: BLE001 - bad DSN / incompatible SDK must never break startup
        logger.warning("Sentry initialisation failed; reporting disabled", exc_info=True)
        return False
    _installed = True
    logger.info(
        "Sentry enabled (environment=%s, release=%s, traces_sample_rate=%s)",
        environment,
        release,
        traces_sample_rate,
    )
    return True


def _reset_for_tests() -> None:
    global _installed
    _installed = False
