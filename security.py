"""Authentication for the control-plane endpoints.

Every state-changing admin endpoint (kill/pause/revive/deploy, provenance seal)
is guarded by a single static API key supplied in the ``X-Admin-Key`` header.
When no key is configured the admin surface is disabled - fail-closed - unless
``ALLOW_INSECURE_DEV=true`` is set for local development.
"""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from config import settings


def require_admin(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")) -> None:
    """Authorize a control-plane request or raise 401/503.

    Used as a FastAPI dependency: ``dependencies=[Depends(require_admin)]``.
    """
    if not settings.admin_api_key:
        if settings.allow_insecure_dev:
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="admin API is disabled (set ADMIN_API_KEY)",
        )

    if x_admin_key is None or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid admin key",
        )
