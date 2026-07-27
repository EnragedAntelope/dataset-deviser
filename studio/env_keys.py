"""Read and edit the API keys in `.env` — the single implementation.

`setup.bat` and `setup.sh` each used to hand-roll this in their own shell dialect,
with different bugs in each: the batch version could not *change* a key that was
already present (a typo was unfixable through the installer) and glued a new key
onto the last line when `.env` lacked a trailing newline; the shell version
appended unconditionally, duplicating every key on a re-run. One of those scripts
also died outright on an unescaped `)` inside an `if` block.

So key handling lives here instead, and the setup scripts call `cli.py keys`. The
shells keep only what they are good at (running pip), the behaviour is identical on
Windows and POSIX, and the fiddly parts — in-place update, newline safety, masking —
are unit-tested.

Secrets rule: values are only ever written to `.env` (0600 on POSIX). Nothing here
prints a whole key; use `mask()` for anything user-visible.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

# Shell/dotenv-safe variable name. Also guards `set_key` against being handed
# something that would write a broken or injected line.
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Matches `NAME=`, `export NAME=`, and tolerates surrounding whitespace, so a
# hand-edited .env is parsed the way python-dotenv would read it.
_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


@dataclass(frozen=True)
class KeySpec:
    """One API key: what it unlocks, what breaks without it, where to get one.

    Single source of truth for the installer, `cli.py doctor`/`keys` and the docs,
    so the three can never drift on what a key is actually for.
    """
    name: str
    unlocks: str
    without: str
    url: str

    @property
    def alias(self) -> str:
        """The `LDS_`-prefixed spelling `Settings` also accepts for this key.

        `.env.example` tells users every setting can be given with the LDS_ prefix,
        and pydantic-settings populates the field from `LDS_*` in preference to the
        bare name — so both spellings must be read, and a write has to land on
        whichever one is actually in the file (see `set_managed_key`).
        """
        return f"LDS_{self.name}"


# ASCII-only text: these strings are printed by setup.bat in a plain cmd window and
# are often redirected to a file, where the app's usual (1)/(2)/(3) circled glyphs and
# em-dashes would raise UnicodeEncodeError on a cp1252/cp437 code page. Stages are
# spelled "stage 2" here rather than "②" for the same reason.
KEY_SPECS: tuple[KeySpec, ...] = (
    KeySpec(
        name="GEMINI_API_KEY",
        unlocks="Cloud image generation (stage 2) and the Gemini captioner (stage 3)",
        without="stage 2 cloud generation and the Gemini captioner are unavailable "
                "(local ComfyUI generation and local captioners still work)",
        url="https://aistudio.google.com/apikey",
    ),
    KeySpec(
        name="GROQ_API_KEY",
        unlocks="Free-tier cloud captioning (stage 3, SFW, rate-limited)",
        without="the Groq captioner is unavailable (other captioners still work)",
        url="https://console.groq.com/keys",
    ),
    KeySpec(
        name="HF_TOKEN",
        unlocks="Built-in SAM3 subject isolation (stage 1, gated model) and Hugging "
                "Face dataset publishing (stage 4)",
        without="stage 1 built-in SAM3 isolation and stage 4 Hugging Face publishing "
                "are unavailable (a ComfyUI SAM3 workflow is an alternative)",
        url="https://huggingface.co/settings/tokens",
    ),
)

KEY_NAMES: tuple[str, ...] = tuple(s.name for s in KEY_SPECS)
KEY_SPECS_BY_NAME: dict[str, KeySpec] = {s.name: s for s in KEY_SPECS}


@dataclass(frozen=True)
class KeyStatus:
    """Whether a key is configured, and a safe-to-print rendering of it."""
    name: str
    is_set: bool
    masked: str
    source: str  # ".env" | "environment" | ""
    spec: KeySpec


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def read_env(path: Path | str) -> dict[str, str]:
    """Parse `NAME=value` pairs from a `.env` file (missing file -> {}).

    Comments, blank lines and non-pair lines are skipped; `export ` prefixes and
    surrounding quotes/whitespace are tolerated, so a hand-edited file reads the
    same way python-dotenv would load it. Never raises on a malformed file.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    pairs: dict[str, str] = {}
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if m:
            pairs[m.group(1)] = _strip_quotes(m.group(2))
    return pairs


_OWNER_ONLY = stat.S_IRUSR | stat.S_IWUSR


def _write_env(path: Path, lines: list[str]) -> None:
    """Write `.env` with a trailing newline, owner-only on POSIX.

    A brand-new file is created 0600 *before* any key is written to it, so a secret
    is never briefly world-readable — the old `touch .env; chmod 600` in setup.sh
    left exactly that window.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt" and not path.exists():
        path.touch(mode=_OWNER_ONLY)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if os.name != "nt":  # also tighten a pre-existing, looser file
        path.chmod(_OWNER_ONLY)


def set_key(path: Path | str, name: str, value: str) -> None:
    """Set `name` to `value` in `.env`, updating in place if it is already there.

    Existing comments, ordering and unrelated settings are preserved. A key that
    is already present is *replaced*, not skipped and not duplicated — so a typo'd
    key can be corrected by re-running setup. Appending to a file that does not end
    in a newline does not glue onto the last line.

    Raises ValueError for a name that is not a valid variable name, or a value
    containing a newline (which would inject a second setting).
    """
    name = name.strip()
    if not _NAME_RE.match(name):
        raise ValueError(f"'{name}' is not a valid environment variable name.")
    value = value.strip()
    if "\n" in value or "\r" in value:
        raise ValueError("An API key cannot contain a line break.")

    path = Path(path)
    existing = path.read_text(encoding="utf-8-sig", errors="replace") if path.is_file() else ""
    lines = existing.splitlines()
    new_line = f"{name}={value}"
    for i, line in enumerate(lines):
        m = _LINE_RE.match(line)
        if m and m.group(1) == name:
            lines[i] = new_line
            break
    else:
        lines.append(new_line)
    _write_env(path, lines)


def unset_key(path: Path | str, name: str) -> None:
    """Remove every line setting `name` from `.env` (no-op if absent)."""
    path = Path(path)
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    kept = [ln for ln in lines
            if not ((m := _LINE_RE.match(ln)) and m.group(1) == name)]
    _write_env(path, kept)


def mask(value: str) -> str:
    """A safe-to-print rendering of a secret: first 4 and last 4 characters only.

    Enough to tell two keys apart or spot a truncated paste, without putting a
    usable secret on screen or in a log. Short values are hidden entirely.
    """
    value = (value or "").strip()
    if not value:
        return "(not set)"
    if len(value) < 12:
        return "*" * 8
    # ASCII "..." rather than "…": this is printed by setup.bat in a plain cmd
    # window and often redirected, where a non-cp1252 glyph can raise
    # UnicodeEncodeError. See studio.doctor.Check.symbol.
    return f"{value[:4]}...{value[-4:]}"


def set_managed_key(path: Path | str, name: str, value: str) -> None:
    """Set one of the `KEY_SPECS` keys, landing on the spelling already in use.

    If the file stores the key under its `LDS_` alias, update *that* line: writing
    the bare name instead would leave the stale alias winning at runtime, so the
    user's "changed" key would silently have no effect. If both spellings are
    present the file is already ambiguous — keep the documented bare name and drop
    the alias.
    """
    spec = KEY_SPECS_BY_NAME.get(name)
    if spec is None:
        set_key(path, name, value)
        return
    present = read_env(path)
    if spec.name in present and spec.alias in present:
        unset_key(path, spec.alias)
        set_key(path, spec.name, value)
    elif spec.alias in present:
        set_key(path, spec.alias, value)
    else:
        set_key(path, spec.name, value)


def unset_managed_key(path: Path | str, name: str) -> None:
    """Remove one of the `KEY_SPECS` keys under either spelling."""
    spec = KEY_SPECS_BY_NAME.get(name)
    unset_key(path, name)
    if spec is not None:
        unset_key(path, spec.alias)


def key_status(path: Path | str = "", *, environ: dict[str, str] | None = None
               ) -> list[KeyStatus]:
    """Configured-or-not for every key in `KEY_SPECS`, safe to print.

    Both the bare name and the `LDS_` alias count, in the order the running app
    actually resolves them: pydantic-settings fills the field from `LDS_*` first,
    so an alias in `.env` is the live value even when the bare name is also there.
    `.env` outranks the process environment as the reported source because that is
    the file the installer and `cli.py keys` edit — but an exported key counts as
    set, since it is equally usable at runtime.
    """
    path = Path(path) if path else _default_env_path()
    environ = os.environ if environ is None else environ
    from_file = read_env(path)
    statuses: list[KeyStatus] = []
    for spec in KEY_SPECS:
        candidates = (
            (from_file.get(spec.alias, ""), f".env ({spec.alias})"),
            (from_file.get(spec.name, ""), ".env"),
            ((environ.get(spec.alias) or ""), f"environment ({spec.alias})"),
            ((environ.get(spec.name) or ""), "environment"),
        )
        value, source = "", ""
        for candidate, label in candidates:
            if candidate.strip():
                value, source = candidate.strip(), label
                break
        statuses.append(KeyStatus(name=spec.name, is_set=bool(value),
                                  masked=mask(value), source=source, spec=spec))
    return statuses


def _default_env_path() -> Path:
    """The repo-root `.env` — the same file `studio.config` loads."""
    from studio.config import REPO_ROOT

    return REPO_ROOT / ".env"


def default_env_path() -> Path:
    """Public accessor for the `.env` the app actually reads."""
    return _default_env_path()
