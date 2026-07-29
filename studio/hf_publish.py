"""Optional: publish a finished dataset folder to the Hugging Face Hub.

Self-contained and strictly opt-in — nothing here runs unless the user asks to
publish. Datasets are created **private by default**; going public is an explicit
choice. The upload authenticates with the token already in the environment
(`HF_TOKEN`, the same one the gated SAM3 download uses); this module never writes
the token anywhere.

`huggingface_hub` is imported lazily inside `publish_dataset`, so import-time cost
is zero for anyone who never publishes. The pure validation (`normalize_repo_id`)
is I/O-free and unit-tested; the network path is exercised only on a real publish.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from studio.config import settings


class HFPublishError(Exception):
    """Publishing cannot proceed (bad repo id, missing token, etc.).

    Plain exception (not gr.Error) so the CLI shares the same logic; the UI
    translates it at the boundary.
    """


# owner/name or a bare name; HF allows letters, digits, '-', '_', '.'.
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


def normalize_repo_id(repo_id: str) -> str:
    """Trim and validate a Hugging Face repo id; raise HFPublishError if invalid."""
    repo_id = (repo_id or "").strip().strip("/")
    if not repo_id:
        raise HFPublishError(
            "Enter a dataset name (e.g. 'my-character-lora' or 'your-user/my-character')."
        )
    if not _REPO_ID_RE.match(repo_id):
        raise HFPublishError(
            f"'{repo_id}' isn't a valid Hugging Face repo id. Use letters, digits, "
            "'-', '_', '.', and at most one '/' (owner/name)."
        )
    return repo_id


def resolve_token(token: str = "") -> str:
    """The explicit token if given, else HF_TOKEN from the environment/.env."""
    return (token or "").strip() or settings.resolved_key("HF_TOKEN")


def _is_existing_public_repo(api, repo_id: str) -> bool:
    """True only when `repo_id` already exists and is public.

    Best-effort: any lookup failure (no such repo, no permission, network) means we
    stay quiet rather than warn wrongly — the warning is a courtesy, not a gate.
    """
    try:
        info = api.dataset_info(repo_id)
    except Exception:
        return False
    return getattr(info, "private", True) is False


def publish_dataset(
    ds_dir: Path | str,
    repo_id: str,
    private: bool = True,
    token: str = "",
    progress: Callable[[str], None] = print,
) -> str:
    """Create (if needed) and upload a dataset folder to the Hub. Returns its URL.

    Private by default. Requires a write token (HF_TOKEN). Only the files inside
    `ds_dir` are uploaded.
    """
    ds_dir = Path(ds_dir)
    if not ds_dir.is_dir():
        raise HFPublishError(f"Dataset folder not found: {ds_dir}")
    repo_id = normalize_repo_id(repo_id)
    tok = resolve_token(token)
    if not tok:
        raise HFPublishError(
            "No Hugging Face token found. Create a WRITE token at "
            "https://huggingface.co/settings/tokens and add it to .env as HF_TOKEN."
        )
    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    if "/" not in repo_id:  # qualify a bare name with the token's own username
        repo_id = f"{api.whoami()['name']}/{repo_id}"
    progress(f"Creating dataset repo {repo_id} ({'private' if private else 'PUBLIC'})...")
    # `exist_ok=True` does NOT change an existing repo's visibility, so re-publishing
    # into a repo that is already public would quietly stay public while the UI said
    # "private". Say so plainly rather than implying a privacy guarantee we can't make.
    if private and _is_existing_public_repo(api, repo_id):
        progress(f"[warning] {repo_id} already exists and is PUBLIC. Uploading to it "
                 "does not make it private — change its visibility in the repo's "
                 "settings on huggingface.co, or publish under a different name.")
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    progress(f"Uploading {ds_dir.name} to {repo_id}...")
    api.upload_folder(folder_path=str(ds_dir), repo_id=repo_id, repo_type="dataset",
                      commit_message="Upload LoRA dataset (LoRA Distillery)")
    url = f"https://huggingface.co/datasets/{repo_id}"
    progress(f"Published: {url}")
    return url
