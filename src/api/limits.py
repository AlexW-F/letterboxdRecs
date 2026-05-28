"""In-process rate limiting + request-size guard for the public API.

Single-instance friendly: per-client-IP fixed-window counters held in
memory. The deployment target (one HF Spaces container) doesn't scale
horizontally, so a shared store (Redis) would be overkill. If this ever
runs multi-replica, swap the in-memory dicts for a shared backend.

Tunables (env):
- ``MAX_UPLOAD_BYTES``        max request body in bytes (default 10 MB)
- ``RATE_WINDOW_SECONDS``     sliding window length (default 60)
- ``RATE_LIMIT_PER_MIN``      requests/window/IP for metered calls (default 60)
- ``RATE_LIMIT_HEAVY_PER_MIN`` requests/window/IP for heavy calls (default 12)
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

MAX_BODY_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
WINDOW_SECONDS = int(os.getenv("RATE_WINDOW_SECONDS", "60"))
GLOBAL_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
HEAVY_LIMIT = int(os.getenv("RATE_LIMIT_HEAVY_PER_MIN", "12"))

# Expensive endpoints: server-side TMDB enrichment, RSS fetch, UMAP render.
HEAVY_PREFIXES = ("/upload-letterboxd", "/explore/personalized")


def _client_ip(request: Request) -> str:
    # HF Spaces / Cloudflare sit in front, so trust the first X-Forwarded-For hop.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:
        super().__init__(app)
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._heavy: Dict[str, Deque[float]] = defaultdict(deque)
        self._last_sweep = time.monotonic()

    def _allow(self, bucket: Dict[str, Deque[float]], ip: str, limit: int, now: float) -> bool:
        dq = bucket[ip]
        cutoff = now - WINDOW_SECONDS
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit:
            return False
        dq.append(now)
        return True

    def _sweep(self, now: float) -> None:
        # Drop idle IPs so the maps don't grow unbounded under IP rotation.
        if now - self._last_sweep < WINDOW_SECONDS:
            return
        cutoff = now - WINDOW_SECONDS
        for bucket in (self._hits, self._heavy):
            for ip in [k for k, dq in bucket.items() if not dq or dq[-1] < cutoff]:
                del bucket[ip]
        self._last_sweep = now

    async def dispatch(self, request: Request, call_next):
        # Cheap size guard from the header before we read the body.
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > MAX_BODY_BYTES:
                    return JSONResponse(
                        {"detail": f"request body too large (max {MAX_BODY_BYTES} bytes)"},
                        status_code=413,
                    )
            except ValueError:
                pass

        now = time.time()
        self._sweep(now)
        path = request.url.path
        is_heavy = any(path.startswith(p) for p in HEAVY_PREFIXES)

        # Meter mutating/expensive calls; let cheap GETs (/health, /modes) pass.
        if request.method != "GET" or is_heavy:
            ip = _client_ip(request)
            if not self._allow(self._hits, ip, GLOBAL_LIMIT, now):
                return JSONResponse(
                    {"detail": "rate limit exceeded; slow down"},
                    status_code=429,
                    headers={"Retry-After": str(WINDOW_SECONDS)},
                )
            if is_heavy and not self._allow(self._heavy, ip, HEAVY_LIMIT, now):
                return JSONResponse(
                    {"detail": "rate limit exceeded for this endpoint; slow down"},
                    status_code=429,
                    headers={"Retry-After": str(WINDOW_SECONDS)},
                )
        return await call_next(request)
