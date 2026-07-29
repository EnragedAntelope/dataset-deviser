"""Best-effort check for a newer GitHub release than the running version.

Never raises and never blocks: any network failure, rate limit, or parse
error just means "no notice shown". Cached 24h like the Gemini model lists
so a normal launch never makes a live call. Disable entirely (no network
call at all) with LDS_UPDATE_CHECK_ENABLED=false.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx

from studio import __version__
from studio.config import CACHE_DIR, settings

UPDATE_CACHE_FILE = CACHE_DIR / "update_check.json"
UPDATE_CACHE_TTL_HOURS = 24
GITHUB_REPO = "EnragedAntelope/lora-distillery"
RELEASES_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RELEASES_PAGE_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"


def _parse_version(v: str) -> tuple[int, ...]:
    """"v0.4.0" / "0.4.0" -> (0, 4, 0); non-numeric segments become 0."""
    parts = []
    for p in v.strip().lstrip("vV").split("."):
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def _is_newer(latest: str, current: str) -> bool:
    try:
        return _parse_version(latest) > _parse_version(current)
    except Exception:
        return False


def _load_cache(*, ignore_ttl: bool = False) -> dict | None:
    if not UPDATE_CACHE_FILE.exists():
        return None
    try:
        data = json.loads(UPDATE_CACHE_FILE.read_text(encoding="utf-8"))
        if not ignore_ttl:
            cached_at = datetime.fromisoformat(data["cached_at"])
            age = datetime.now(tz=timezone.utc) - cached_at
            if age > timedelta(hours=UPDATE_CACHE_TTL_HOURS):
                return None
        return data
    except Exception:
        return None


def _save_cache(latest: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    UPDATE_CACHE_FILE.write_text(
        json.dumps({"cached_at": datetime.now(tz=timezone.utc).isoformat(),
                    "latest": latest}, indent=2),
        encoding="utf-8",
    )


def _fetch_latest_tag() -> str | None:
    try:
        r = httpx.get(RELEASES_API_URL, timeout=5,
                      headers={"Accept": "application/vnd.github+json",
                               "User-Agent": "lora-distillery-update-check"})
        r.raise_for_status()
        tag = r.json().get("tag_name", "")
        return tag or None
    except Exception:
        return None


def latest_version(force_refresh: bool = False) -> str | None:
    """Latest published release tag (e.g. "v0.4.0"), via a 24h cache.

    Falls back to a stale cached value if a refresh is due but GitHub is
    unreachable; returns None only if nothing has ever been fetched.
    """
    if not force_refresh:
        fresh = _load_cache()
        if fresh is not None:
            return fresh.get("latest")
    tag = _fetch_latest_tag()
    if tag:
        _save_cache(tag)
        return tag
    stale = _load_cache(ignore_ttl=True)
    return stale.get("latest") if stale else None


def check_for_update(force_refresh: bool = False) -> dict | None:
    """{"current", "latest", "url"} if a newer release is published, else None."""
    if not settings.update_check_enabled:
        return None
    tag = latest_version(force_refresh=force_refresh)
    if not tag or not _is_newer(tag, __version__):
        return None
    return {"current": __version__, "latest": tag.lstrip("vV"), "url": RELEASES_PAGE_URL}


def update_banner_markdown(force_refresh: bool = False) -> str:
    """Markdown for the UI banner, or "" if no update / check disabled."""
    info = check_for_update(force_refresh=force_refresh)
    if not info:
        return ""
    return (
        f"🔔 **Update available:** v{info['current']} → v{info['latest']} — "
        f"[see what's new]({info['url']})."
    )
