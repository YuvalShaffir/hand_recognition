# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Webcam hand tracking (MediaPipe) that lets you record a hand gesture as a
named "macro" and, when the same gesture is performed again live, fires a
mapped OS mouse action (left-click, right-click, scroll, ...) via
`pyautogui`. It can also drive the real OS cursor directly from the hand's
position ("cursor mode"), independently of and simultaneously with gesture
matching.

## Running

`hand_recognition` (`src/hand_recognition/`) is the real application —
**not** `main.py` at the repo root, which is an unrelated bare-webcam
leftover prototype. It must run under Windows-native `uv` (this machine: a
`uv`/venv on the Windows side of the filesystem, invoked from Windows
PowerShell in this same repo directory), not the WSL `.venv`, because
`pyautogui` needs to control the actual Windows desktop and the WSL side
generally can't reach the real cursor/webcam. Both sides share this same
`pyproject.toml`/`uv.lock` — `uv` just builds a separate, OS-native `.venv`
on each side. See `docs/ARCHITECTURE.md` for the full reasoning and data
flow.

From a Windows PowerShell prompt, in the repo root:

```
uv run hand-recognition
```

(equivalently `uv run python -m hand_recognition`; running
`src/hand_recognition/__main__.py` directly as a script path does **not**
work — its relative imports require it to be executed as a package via
`-m` or the console-script entry point).

On first run, `hand_landmarker.task` (the MediaPipe model, ~8MB) auto-downloads
to the repo root via `vision.ensure_model()` if not already present — it's
gitignored (generated binary asset, not committed), so a fresh clone needs
network access once. Same for `recordings/` — starts empty, populated by `r`.

In the running window: `r` toggles recording a gesture (stop prompts for an
action name in the terminal, e.g. `left-click`); `c` toggles cursor mode
(moves the real OS cursor with the hand); `[`/`]` tighten/loosen the gesture
match threshold live (shown on-screen as `thresh: N.NN`) to cut down false
positives; `q` quits. Recordings are saved to `recordings/<name>.npz`. Known
action names live in `src/hand_recognition/actions.py`'s
`ActionDispatcher` — an unrecognized name still records and can be matched,
but won't fire anything.

All of the above tunables (paths, camera settings, detection confidences,
quantization bin/hysteresis, match threshold bounds, cursor smoothing,
scroll amount, keybindings) live in `config.json` at the repo root — see
`src/hand_recognition/config.py` for the schema and defaults. Missing keys
fall back to defaults; unknown keys raise at load time.

## Commands (WSL, via `uv`)

This repo lives on the Windows `E:` drive (mounted at `/mnt/e` in WSL), so
WSL and Windows share the same directory — including, by default, the same
`.venv`. A WSL-built venv and a Windows-built venv aren't binary-compatible
(different `mediapipe`/`opencv-python` wheels, and WSL venvs include a
`lib64` symlink Windows can't clean up), so **WSL must use a separate venv
path**, `.venv-wsl` (gitignored), via `UV_PROJECT_ENVIRONMENT`:

```
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run black .
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run flake8
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run mypy .
```

(Exporting `UV_PROJECT_ENVIRONMENT=.venv-wsl` once per WSL shell session in
this repo avoids repeating it on every command.) Leave Windows-side `uv`
commands (`uv sync`, `uv run hand-recognition`) alone — they should use the
default `.venv`. These WSL commands check the code but can't run the app
end-to-end (see above). No automated test suite exists yet.

## Code shape

Everything between the camera and the OS is a `Stage` (`stage.py`): one item
in, one item out, composed with `|`. Two rules follow from that and are load-
bearing — break either and the bugs are subtle:

- **A stage that has nothing to report returns `None`; it never skips.**
  `OptionalStage` passes `None` through untouched. This keeps every stage
  one-item-per-camera-frame, so the two branches of the `Fork` can't drift
  out of step with each other.
- **A stage works both per-item and per-stream.** `apply(item)` for one item,
  calling it on an iterator for a stream. The desktop app pulls (a `for` loop
  over the camera); Streamlit pushes (a callback per frame). Adding a stage
  that only supports one of those breaks a front-end.

`domain.py` holds the types that travel between stages; items carry their own
`timestamp_ms` rather than the pipeline threading a clock alongside them.
New work goes in the folder for its link in the chain — `vision/`,
`gestures/`, `cursor/` — with `apps/` holding front-ends that own no
recognition state.

The Streamlit demo runs with `streamlit run src/hand_recognition/apps/web.py`
(this one *does* work from WSL — it's report-only and never imports
`pyautogui`).

## Architecture & design

- `CONTEXT.md` — the project's vocabulary. Use these words as written;
  the "_Avoid_" lines are the near-misses to keep out of the code.
- `docs/ARCHITECTURE.md` — module map, data flow, the stage model, the
  two-Python-environment split, the `.npz` recording format.
- `docs/DESIGN_MATH.md` — why joint angles (not raw coordinates), why
  `hand_world_landmarks` (not `hand_landmarks`), the quantization/hysteresis
  scheme, the DTW matching cost function, and the cursor-mode
  smoothing/dead-zone scheme.
- `docs/PYTHON_STANDARDS.md` — code style conventions used in this repo.
- `docs/TEST_PLAN.md` — the planned `pytest` suite. No suite exists yet.
- `docs/adr/` — decisions with consequences, one file each.

## Agent skills

### Issue tracker

Issues live as GitHub issues in `YuvalShaffir/hand_recognition`, managed with
the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, used verbatim as label strings. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See
`docs/agents/domain.md`.
