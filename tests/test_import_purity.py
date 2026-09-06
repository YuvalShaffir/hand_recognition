"""`web.py` must not import `actions.py` or `cursor.driver`: both set
`pyautogui.FAILSAFE` at module scope, and importing `pyautogui` fails
outright on a headless container. Nothing checked it before this file."""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import hand_recognition

PACKAGE_ROOT = Path(hand_recognition.__file__).parent

RECOGNITION_MODULES = [
    "hand_recognition.config",
    "hand_recognition.domain",
    "hand_recognition.stage",
    "hand_recognition.gestures",
    "hand_recognition.gestures.angles",
    "hand_recognition.gestures.library",
    "hand_recognition.gestures.matcher",
    "hand_recognition.gestures.movement",
    "hand_recognition.gestures.persistence",
    "hand_recognition.gestures.pipeline",
    "hand_recognition.gestures.recorder",
    "hand_recognition.vision",
    "hand_recognition.vision.capture",
    "hand_recognition.vision.detection",
    "hand_recognition.vision.model_asset",
    "hand_recognition.cursor",
    "hand_recognition.cursor.center",
    "hand_recognition.cursor.pipeline",
    "hand_recognition.cursor.screen",
]

_BLOCK_PYAUTOGUI = """
import sys


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "pyautogui" or name.startswith("pyautogui."):
            raise ImportError("pyautogui is unavailable here")
        return None


sys.modules.pop("pyautogui", None)
sys.meta_path.insert(0, Blocker())
"""


def run_python(*parts: str) -> subprocess.CompletedProcess:
    script = "\n".join(textwrap.dedent(part) for part in parts)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_recognition_modules_import_without_pyautogui():
    imports = "\n".join(
        f"importlib.import_module({module!r})" for module in RECOGNITION_MODULES
    )
    result = run_python(
        _BLOCK_PYAUTOGUI,
        "import importlib",
        imports,
        'print("imported")',
    )

    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


def test_the_pyautogui_blocker_is_not_vacuous():
    """Without this, the test above would pass on an environment where
    `pyautogui` imports fine, proving nothing."""
    result = run_python(
        _BLOCK_PYAUTOGUI,
        """
        import importlib

        try:
            importlib.import_module("hand_recognition.cursor.driver")
        except ImportError:
            print("blocked")
        """,
    )

    assert result.returncode == 0, result.stderr
    assert "blocked" in result.stdout


def _module_path(module: str) -> Path | None:
    relative = Path(*module.split(".")[1:])
    for candidate in (
        PACKAGE_ROOT / relative.with_suffix(".py"),
        PACKAGE_ROOT / relative / "__init__.py",
    ):
        if candidate.exists():
            return candidate
    return None


def _imported_modules(module: str, path: Path) -> set[str]:
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                prefix = package.split(".")
                base = ".".join(
                    prefix[: len(prefix) - node.level + 1] + ([base] if base else [])
                )
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return {name for name in found if name.startswith("hand_recognition")}


def module_graph(entry: str) -> set[str]:
    """Every `hand_recognition` module reachable from `entry`, read rather
    than executed - importing Streamlit is not this test's business."""
    seen: set[str] = set()
    queue = [entry]
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        path = _module_path(module)
        if path is None:
            continue
        queue.extend(_imported_modules(module, path) - seen)
    return seen


@pytest.mark.parametrize(
    "forbidden",
    ["hand_recognition.actions", "hand_recognition.cursor.driver"],
)
def test_web_app_module_graph_excludes_pyautogui(forbidden):
    assert forbidden not in module_graph("hand_recognition.apps.web")


def test_the_module_graph_walker_is_not_vacuous():
    assert "hand_recognition.actions" in module_graph("hand_recognition.apps.desktop")


def test_cursor_package_init_does_not_pull_the_driver():
    result = run_python("""
        import sys
        import hand_recognition.cursor
        print("hand_recognition.cursor.driver" in sys.modules)
        """)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
