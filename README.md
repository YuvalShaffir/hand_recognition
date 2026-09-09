# hand_recognition

[![CI](https://github.com/YuvalShaffir/hand_recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/YuvalShaffir/hand_recognition/actions/workflows/ci.yml)

Webcam hand-gesture recognition that maps custom gestures to real OS mouse
actions — record a hand pose sequence as a named macro, and when you perform
it again live, it fires the mapped action (`left-click`, `right-click`,
`scroll-up`, ...) via `pyautogui`. It can also drive the OS cursor directly
from hand position ("cursor mode"), simultaneously with gesture matching.

Built on [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
hand landmark detection.

## How it works

```
camera frame -> MediaPipe HandLandmarker -> 21 3D hand landmarks
             |
             +-> 15 joint angles (finger bend/spread, in degrees)
             |   -> quantized to angle bins with hysteresis (anti-jitter)
             |   -> event-driven sequence of pose changes
             |   -> DTW-matched live against recorded gesture templates
             |   -> best match above threshold -> mapped OS action fires
             |
             +-> hand centre point -> dead zone -> screen remap -> EMA
                 -> the real OS cursor moves
```

Each arrow is a `Stage`: one item in, one item out, composed with `|` into
the two pipelines above. Both run off the same detected hand every frame,
neither aware of the other.

A few choices make this work reliably:

- **Joint angles, not raw coordinates.** Each gesture is reduced to 15 angles
  (3 per finger) computed from MediaPipe's metric 3D landmarks. Angles are
  translation/scale/rotation-invariant, so the same gesture matches
  regardless of where the hand is in frame, how close to the camera, or its
  orientation.
- **Quantization with hysteresis.** Angles are snapped to bins (default 15°)
  using a Schmitt-trigger scheme so landmark jitter near a bin boundary
  doesn't flicker between bins every frame.
- **Event-driven recording.** Only bin *changes* are recorded, collapsing a
  30fps stream of near-identical frames into the actual sequence of distinct
  key poses (typically 5-20 per gesture) — cheap to store and cheap to match.
- **DTW matching.** Live pose-change events are matched against recorded
  templates with dynamic time warping, so the same gesture performed faster
  or slower still matches, while poses hit in the wrong order don't.
  Threshold is adjustable live (`[`/`]`) to trade off false positives vs.
  false negatives while testing.
- **Cursor mode runs independently.** Cursor position tracks the hand
  centroid (not a fingertip, which would jump around as fingers move through
  a gesture), with a dead zone + EMA smoothing so it sits still when the hand
  is still. This is what lets cursor control and gesture matching run at the
  same time off the same hand.

Full design rationale: [`docs/DESIGN_MATH.md`](docs/DESIGN_MATH.md).
Module map and data flow: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Setup & running

Requires [`uv`](https://docs.astral.sh/uv/), and a **Windows-native shell**
— PowerShell, or a genuine Windows Git Bash. Not WSL, even if your checkout
happens to live on a WSL-mounted drive: `pyautogui` needs to control the
real desktop, and only a Windows-native Python can reach that (or the real
webcam).

From that Windows-native shell, clone and run in one line:

```
git clone https://github.com/YuvalShaffir/hand_recognition.git && cd hand_recognition && uv run hand-recognition
```

`uv run` installs everything (`mediapipe`, `opencv-python`, `pyautogui`,
...) into a local `.venv` on first use — no separate install step needed,
and safe to re-run any time after (it's a no-op once dependencies are
already in sync). The hand-tracking model (`hand_landmarker.task`, ~8MB)
also downloads automatically on first run, since it's a generated binary
asset and isn't committed to the repo. A window titled `camera` then opens
showing your webcam feed with the hand skeleton overlaid.

Once cloned, day-to-day runs are just:

```
uv run hand-recognition
```

> **Developing from WSL?** Linting/type-checking (`black`/`flake8`/`mypy`)
> can run from WSL, but must use a separate venv path so it doesn't collide
> with the Windows one sharing the same directory:
> `UV_PROJECT_ENVIRONMENT=.venv-wsl uv run black .`. See `CLAUDE.md` for
> the rest of the WSL/Windows split.

### Configuration

Every tunable — camera index/resolution, detection-confidence thresholds,
match-threshold bounds, quantization bin size, cursor smoothing/deadzone,
scroll amount, keybindings — lives in [`config.json`](config.json) at the
repo root. Edit it and re-run; no code changes needed. A missing file or
omitted key falls back to the shipped default; an unknown key raises an
error at startup (typo protection). Full schema:
[`src/hand_recognition/config.py`](src/hand_recognition/config.py).

### Controls

| key | action |
|-----|--------|
| `r` | toggle recording a gesture (stop prompts for an action name in the terminal, e.g. `left-click`) |
| `c` | toggle cursor mode (moves the real OS cursor with the hand) |
| `[` / `]` | tighten / loosen the gesture match threshold live |
| `q` | quit |

(Keybindings above are the defaults — remappable in `config.json`.)
Recordings save to `recordings/<name>.npz`. Known action names live in
[`src/hand_recognition/actions.py`](src/hand_recognition/actions.py)'s
`ActionDispatcher` — an unrecognized name still records and can be matched,
but won't trigger anything.

## Project layout

| module | role |
|--------|------|
| `stage.py` | `Stage`/`OptionalStage`/`Fork` — the composition primitives |
| `domain.py` | the types that travel between stages |
| `vision/` | webcam -> mirrored, timestamped frames -> detected hands |
| `gestures/` | hand -> pose -> movement -> gesture name, plus recording and the template library |
| `cursor/` | hand -> centre point -> screen position -> real OS cursor |
| `actions.py` | gesture name -> `pyautogui` action |
| `apps/` | the two front-ends: desktop (`desktop.py`) and Streamlit (`web.py`), plus shared `overlay.py` drawing |
| `config.py` | `AppConfig` dataclasses + `config.json` loading |
| `__main__.py` | entry point (`python -m hand_recognition`) |

All under `src/hand_recognition/`. Full data flow and module details:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the vocabulary these names
come from: [`CONTEXT.md`](CONTEXT.md).

There is also a browser demo — the same pipelines, reporting matches instead
of driving the OS:

```
streamlit run src/hand_recognition/apps/web.py
```

It never imports `pyautogui`, so unlike the desktop app it runs anywhere,
WSL included.

Deploying it to Streamlit Community Cloud needs `packages.txt` at the repo
root: `opencv-python` links against `libGL`/`glib`, which the deploy image
does not ship, and `import cv2` fails at startup without them.
