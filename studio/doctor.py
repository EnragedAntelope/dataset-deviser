"""Environment self-check behind `cli.py doctor`, plus the `cli.py keys` helpers.

Why this exists: the only symptom of a broken install used to be a console window
closing. `doctor` turns that into a plain statement of what is present, what is
missing, and what each missing piece blocks — so a confused user (or a maintainer
after a `git pull` that added a dependency) gets an answer instead of silence.

Every check takes its inputs as parameters (`importer`, `version`, `environ`) rather
than reading the live interpreter, so the whole report is unit-testable without
mutating the process. Nothing here prints a whole secret — keys go through
`env_keys.mask`.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from studio import __version__, env_keys

# The lowest Python we support, and the highest we have actually tested against.
# Above MAX we warn rather than block: it usually works, but wheels for torch /
# onnxruntime often lag a new release, and that failure is otherwise baffling.
MIN_PYTHON = (3, 10)
MAX_TESTED_PYTHON = (3, 13)

# Import names (not pip names) that `app.py` needs at startup. A dependency added
# to requirements.txt but missing here would reintroduce the exact failure the
# maintainer hit: `start.bat` dying on an import error after a pull.
REQUIRED_IMPORTS: tuple[str, ...] = (
    "gradio",
    "pandas",
    "numpy",
    "PIL",
    "httpx",
    "typer",
    "pydantic",
    "pydantic_settings",
    "dotenv",
    "yaml",
    "huggingface_hub",
)

Importer = Callable[[str], bool]


def _can_import(name: str) -> bool:
    """True if `name` is importable, without importing it (keeps `doctor` fast)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class Check:
    """One line of the report. `warn` means advisory — it does not fail the run."""
    label: str
    ok: bool
    detail: str = ""
    warn: bool = False

    @property
    def symbol(self) -> str:
        # ASCII on purpose: this report is printed by setup.bat inside a plain cmd
        # window and is often redirected to a file or piped. On a machine whose ANSI
        # code page is cp1252/cp437 (i.e. most Windows installs without the UTF-8
        # locale option), writing "✓" to a pipe raises UnicodeEncodeError — and a
        # diagnostic that crashes while diagnosing is worse than a plain one.
        if not self.ok:
            return "[FAIL]"
        return "[warn]" if self.warn else "[ ok ]"


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    key_lines: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """False only for real failures; warnings and missing keys don't count."""
        return all(c.ok for c in self.checks)


# --- individual checks -----------------------------------------------------

def check_python(version: tuple[int, ...]) -> Check:
    """Verify the running interpreter is in the supported range."""
    shown = ".".join(str(p) for p in version[:3])
    if version[:2] < MIN_PYTHON:
        need = ".".join(str(p) for p in MIN_PYTHON)
        return Check("Python", False,
                     f"{shown} is too old - this project needs Python {need} or newer. "
                     "Install it from https://www.python.org/downloads/, then delete the "
                     ".venv folder and re-run setup.")
    if version[:2] > MAX_TESTED_PYTHON:
        tested = ".".join(str(p) for p in MAX_TESTED_PYTHON)
        return Check("Python", True,
                     f"{shown} (newer than the tested {tested} - if pip cannot find a "
                     "torch or onnxruntime wheel, use the tested version instead)",
                     warn=True)
    return Check("Python", True, shown)


def check_venv(executable: str = "") -> Check:
    """Report whether the app is running from the project's own virtualenv."""
    exe = executable or sys.executable
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        return Check("Virtualenv", True, exe)
    return Check("Virtualenv", True,
                 f"{exe} (not a virtualenv - fine if you installed the requirements "
                 "globally, but setup.bat/setup.sh create .venv for you)", warn=True)


def check_dependencies(importer: Importer | None = None) -> Check:
    """Verify every module `app.py` imports at startup is installed."""
    imp = importer or _can_import
    missing = [name for name in REQUIRED_IMPORTS if not imp(name)]
    if not missing:
        return Check("Dependencies", True, f"all {len(REQUIRED_IMPORTS)} present")
    return Check("Dependencies", False,
                 f"missing: {', '.join(missing)} - re-run setup.bat (Windows) or "
                 "./setup.sh (Linux/macOS) to install them. This is what happens after "
                 "a `git pull` that adds a dependency.")


def check_onnxruntime(importer: Importer | None = None) -> Check:
    """Optional: only the WD/Z3D taggers (stage 3) need it, and it is lazy-imported."""
    imp = importer or _can_import
    if imp("onnxruntime"):
        return Check("onnxruntime", True, "installed (WD/e621 taggers available)")
    return Check("onnxruntime", True,
                 "not installed - only the stage 3 WD/e621 taggers need it. Re-run "
                 "setup, or `pip install onnxruntime` (CPU) / `onnxruntime-gpu` (CUDA).",
                 warn=True)


def check_torch(importer: Importer | None = None) -> Check:
    """Advisory: local generation/captioning/isolation need torch; cloud does not."""
    imp = importer or _can_import
    if imp("torch"):
        return Check("PyTorch", True, "installed (local models available)")
    return Check("PyTorch", True,
                 "not installed - local captioners, built-in SAM3 isolation and local "
                 "generation are unavailable. Cloud options still work. Re-run setup to "
                 "install it.", warn=True)


def check_comfyui() -> Check:
    """Optional local generation backend; unreachable is normal and fine."""
    try:
        from studio import comfy_api

        up = comfy_api.is_up()
    except Exception:
        up = False
    from studio.config import settings

    if up:
        return Check("ComfyUI", True, f"reachable at {settings.comfy_url}")
    return Check("ComfyUI", True,
                 f"not reachable at {settings.comfy_url} - optional. Cloud generation and "
                 "built-in SAM3 isolation work without it (see docs/comfyui-setup.md).",
                 warn=True)


def check_comfyui_models() -> Check | None:
    """If ComfyUI is reachable, validate every bundled template's configured model
    filename against the server's own list.

    Reuses `comfy_api.load_template()` itself — the exact seam a real generate/
    preprocess run goes through — rather than re-implementing the matching, so
    this can never drift from what actually happens at run time. Returns None
    (skip entirely) when ComfyUI is unreachable; `check_comfyui()` already
    reports that case.
    """
    from studio import comfy_api

    try:
        if not comfy_api.is_up():
            return None
    except Exception:
        return None

    problems: list[str] = []
    for path in sorted(comfy_api.WORKFLOWS_DIR.glob("*.json")):
        try:
            comfy_api.load_template(path.stem)
        except comfy_api.ComfyError as e:
            problems.append(str(e))
        except Exception:
            continue  # a template JSON/parse issue isn't this check's job to report
    if not problems:
        return Check("ComfyUI models", True, "all configured model filenames found on the server")
    return Check("ComfyUI models", True, " | ".join(problems), warn=True)


# --- keys ------------------------------------------------------------------

def key_report(env_path: Path | str = "", *, environ: dict[str, str] | None = None
               ) -> list[str]:
    """Masked, safe-to-print status for every key, naming what each one blocks."""
    lines: list[str] = []
    for status in env_keys.key_status(env_path, environ=environ):
        if status.is_set:
            lines.append(f"  [ ok ] {status.name:<16} set  {status.masked}  "
                         f"(from {status.source})")
            lines.append(f"         unlocks: {status.spec.unlocks}")
        else:
            lines.append(f"  [ -- ] {status.name:<16} not set")
            lines.append(f"         without it: {status.spec.without}")
            lines.append(f"         get one at: {status.spec.url}")
    return lines


def prompt_secret(label: str) -> str:
    """Read a secret without echoing it, or fall back when there is no terminal.

    `getpass` on Windows reads the console device directly (`msvcrt.getwch`) rather
    than stdin, so with stdin redirected — a piped setup run, a CI job, some
    launchers — it blocks forever waiting for a keypress nobody can send. Detect
    that and read the line instead, warning that the input will be visible. A
    hidden prompt is the goal; hanging is not an acceptable way to achieve it.
    """
    import getpass

    stdin = getattr(sys, "stdin", None)
    try:
        interactive = bool(stdin is not None and stdin.isatty())
    except (AttributeError, ValueError):  # closed or replaced stream
        interactive = False
    if interactive:
        return getpass.getpass(label)
    print(f"{label}(no terminal detected - input will be VISIBLE) ", end="", flush=True)
    return input()


# Obvious placeholders users paste from a docs page instead of a real key.
_PLACEHOLDERS = {
    "your_api_key_here", "your-api-key-here", "yourapikey", "your_key_here",
    "xxx", "xxxx", "changeme", "todo", "none", "null", "paste_your_key_here",
}


def apply_key_input(env_path: Path | str, name: str, raw: str) -> bool:
    """Apply one prompt answer. Returns True if `.env` changed.

    Empty input keeps whatever is already there (so Enter is always the safe
    answer on a re-run); a single `-` clears the key. Wrapping quotes are stripped
    because users paste them, and obvious placeholder text is rejected rather than
    silently saved as a broken key.
    """
    value = (raw or "").strip()
    if not value:
        return False
    if value == "-":
        env_keys.unset_managed_key(env_path, name)
        return True
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    if value.lower() in _PLACEHOLDERS:
        raise ValueError(
            f"'{raw.strip()}' looks like placeholder text, not a real key. "
            f"Get one at {env_keys.KEY_SPECS_BY_NAME[name].url}, or press Enter to skip."
        )
    env_keys.set_managed_key(env_path, name, value)
    return True


# --- assembly --------------------------------------------------------------

def build_report(env_path: Path | str = "", *, environ: dict[str, str] | None = None,
                 importer: Importer | None = None,
                 version: tuple[int, ...] | None = None,
                 check_comfy: bool = True) -> Report:
    """Run every check and collect the results (no I/O beyond the checks)."""
    report = Report()
    report.checks.append(check_python(version or sys.version_info[:3]))
    report.checks.append(check_venv())
    report.checks.append(check_dependencies(importer))
    report.checks.append(check_torch(importer))
    report.checks.append(check_onnxruntime(importer))
    if check_comfy:
        report.checks.append(check_comfyui())
        models_check = check_comfyui_models()
        if models_check is not None:
            report.checks.append(models_check)
    report.key_lines = key_report(env_path, environ=environ)
    return report


def render(report: Report) -> str:
    """The human-readable report. Ends with the command that fixes a missing key.

    Deliberately ASCII-only throughout — see `Check.symbol` for why.
    """
    out = [f"Dataset Deviser v{__version__} - environment check", ""]
    for check in report.checks:
        out.append(f"  {check.symbol} {check.label:<14} {check.detail}")
    out += ["", "API keys (all optional - every stage has a local path):"]
    out += report.key_lines
    out += ["", "Set or change a key:  python cli.py keys --set GEMINI_API_KEY",
            "Review them all:      python cli.py keys"]
    if not report.ok:
        out += ["", "FAIL: something above needs fixing before the app will start."]
    else:
        out += ["", "OK: ready - launch with start.bat (Windows) or ./start.sh."]
    return "\n".join(out)
