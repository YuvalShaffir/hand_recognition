# Architecture

## Entry point

`src/hand_recognition/` is the real application, an installable package
(`pyproject.toml` declares it via `hatchling`, src-layout auto-detected from
the project name). `main.py` at the repo root is a **leftover bare-webcam
prototype** (opens a camera, shows the frame, no hand tracking or gesture
logic) — despite the name, it is not part of the pipeline, isn't part of
the package, and isn't kept in sync with the rest of the code.

Within the package, `__main__.py` is a thin entry point
(`DesktopApp().run()`), reached via `python -m hand_recognition` or the
`hand-recognition` console script.

Three modules sit at the package root because everything else is written
in terms of them:

- `stage.py` — `Stage`, the one-item-in/one-item-out building block every
  step is made of, plus `OptionalStage` (passes `None` through) and `Fork`
  (feeds one stream to two stages). Stages compose with `|`.
- `domain.py` — the types that travel between stages (`Frame`, `Hand`,
  `Detection`, `Pose`, `Movement`, `ScreenPoint`, `GestureTemplate`).
- `config.py` — `AppConfig` dataclass tree + `load_config()`; see
  "Configuration" below.

Then one folder per link in the chain, each a pipeline over those:

- `vision/` — frames and hands. `capture.py` (`CaptureManager`: the webcam
  as a stream of mirrored, timestamped `Frame`s), `detection.py`
  (`HandDetector`, a `Stage[Frame, Detection]` wrapping MediaPipe's
  `HandLandmarker` plus a thread-safe holder for the latest async result),
  and `model_asset.py` (`ensure_model()`, the first-run model download).
- `gestures/` — hands to gesture names (see the module map below).
- `cursor/` — hands to screen positions (same).
- `actions.py` — gesture names to OS effects.

And the front-ends, which own no recognition state:

- `apps/desktop.py` — `DesktopApp`: the desktop program (window, keyboard,
  OS output), which wires the camera to the pipelines. This is what used to
  be the monolithic `app.py`.
- `apps/web.py` — the Streamlit demo, `streamlit run
  src/hand_recognition/apps/web.py`. Report-only: it must not reach
  `actions.py` or `cursor/driver.py`, which import `pyautogui`.
- `apps/overlay.py` — frame drawing shared by both: hand skeleton
  (`draw_landmarks`) and HUD text (`draw_hud`,
  `draw_recording_prompt_hint`).

`gesture_recognizer.task` (repo root) is an unused leftover asset — no code
currently loads it. Only `hand_landmarker.task` is used, by `vision/detection.py`.

Neither `.task` file is committed (`*.task` is gitignored — they're
generated binary assets, ~8MB each). `vision.ensure_model()` downloads
`hand_landmarker.task` from MediaPipe's official hosted URL on first run if
it's missing, so a fresh clone only needs `uv run hand-recognition` (plus
one-time network access) — no manual asset step.

## The stage model

Everything between the camera and the OS is a `Stage`: one item in, one item
out, composed left to right with `|` into a longer stage that is itself a
stage. `GesturePipeline` and `CursorPipeline` are exactly that — a composed
chain plus the small API the front-ends need (recording, threshold, on/off)
— which is why a front-end can hold one and never see the steps inside it.

Two consequences are worth stating outright, because they explain shapes
that would otherwise look odd:

**Nothing-to-report is `None`, not a skipped item.** A stage that only
produces a result some of the time — no hand in frame, hand held still,
gesture not matched — returns `None`, and `OptionalStage` passes `None`
straight through without calling the step's own logic. If instead a stage
could drop items, the two branches of the fork would advance at different
rates and a "current" cursor position could belong to a different frame than
the gesture beside it. One item per frame, all the way down, removes that
class of bug entirely.

**Stages are both push and pull.** `apply(item)` handles a single item;
calling the stage on an iterator returns an iterator. The desktop app pulls
— it drives a `for` loop over the camera — while Streamlit pushes, handing
one frame at a time to a callback on its own worker thread. The same stage
objects serve both without a shim.

`Fork` runs its two branches one after the other, not concurrently. They are
genuinely independent and only depend on the shared input, so threading them
is tempting, and it was measured: it came out slower (0.882 vs 0.805
ms/frame). The cursor branch is about a hundredth of the gesture branch's
cost, so there is almost nothing to overlap, the pure-Python DTW loop holds
the GIL regardless, and the submit/join handoff costs more than the overlap
saves. Sequential is both faster and simpler here.

## Configuration

Every tunable (paths, camera settings, detection confidences, quantization
bin/hysteresis, match threshold bounds/cooldown, cursor smoothing/deadzone,
action scroll amount, keybindings) is a field on `AppConfig`
(`src/hand_recognition/config.py`), grouped into per-concern sections
(`PathsConfig`, `CameraConfig`, `LandmarkerConfig`, `QuantizeConfig`,
`MatchConfig`, `CursorConfig`, `ActionsConfig`, `KeybindConfig`).

`load_config()` reads `config.json` from the current working directory
(same relative-path convention as `hand_landmarker.task` and `recordings/`
— run from the repo root), merges it over per-field defaults, and raises
`ValueError` on any section or key the dataclasses don't define, so a typo
in `config.json` fails loudly instead of silently no-opping. A missing
`config.json` yields `AppConfig()` (all defaults, matching the checked-in
`config.json`). `DesktopApp()` calls `load_config()` itself when constructed
without an explicit `AppConfig`, then threads the relevant section into
each pipeline's constructor (e.g. `CursorPipeline(config.cursor,
screen_size)`).

## Runtime split (why two `uv` venvs, not two Pythons)

This runs across two OS-native `uv` environments sharing the same
`pyproject.toml`/`uv.lock` — `uv` builds a separate venv per OS since
compiled deps (`opencv-python`, `mediapipe`) aren't portable across them:

- **WSL `.venv-wsl`** — editing, `black`/`flake8`/`mypy`. WSL has no direct
  access to the real Windows desktop/cursor or (typically) the webcam, so
  `pyautogui` calls made from here can't reach the real desktop (importing
  it here also hits an unrelated `Xlib.error.XauthError` — no
  `~/.Xauthority` — since there's no X server).
- **Windows-native `.venv`** — where the app actually runs, via
  `uv run hand-recognition` from a Windows PowerShell prompt in the repo
  root. `pyautogui` must run in the same OS instance as the desktop it's
  clicking on, which is why mouse actions can't be driven from the WSL
  side.

**Why WSL uses `.venv-wsl` instead of the default `.venv`:** this repo lives
on the Windows `E:` drive, mounted at `/mnt/e` in WSL — so both sides
operate on the literal same directory, including whatever's at `.venv`. A
WSL-built venv isn't binary-compatible with a Windows one, and WSL venvs
contain a `lib64` symlink that Windows `uv` cannot remove to rebuild the
directory (`Access is denied`). Pointing WSL at `.venv-wsl` via
`UV_PROJECT_ENVIRONMENT=.venv-wsl` (both gitignored) keeps the two
completely separate. Both are still built from the same lockfile — no
manual/global `pip` installs on either side.

## Data flow

Every stage is one-item-in/one-item-out, so nothing falls out of step with
the frames driving it: a stage with nothing to report yields `None`, and
`None` passes through the rest of the pipeline untouched.

```
CaptureManager                             (vision/capture.py)
  -> Frame (mirrored, strictly increasing timestamp)
HandDetector                               (vision/detection.py)
  -> HandLandmarker.detect_async(); the callback stores, under a lock,
     the latest Hand(s) - each holding two views of the same hand:
       landmarks        - image-space (pixel drawing, cursor mode)
       world_landmarks  - real-world metric 3D (gesture features)
  -> Detection(frame, hands); `.primary` is the hand the pipelines read

Fork(GesturePipeline, CursorPipeline).apply(detection.primary)
                                           (apps/desktop.py)

GesturePipeline = AngleExtractor | MovementExtractor              (gestures/)
                | GestureRecorder | GestureMatcher
  -> AngleExtractor    Hand -> Pose: 21 world landmarks -> 15 joint angles
  -> MovementExtractor Pose -> Movement, emitted only when the quantized
                       pose crosses a bin (hysteresis); a still hand
                       yields None (event-driven -> run-length-collapsed)
  -> GestureRecorder   while recording, collects the movements and passes
                       None on, so a gesture can't fire as it is recorded
  -> GestureMatcher    DTW-aligns a rolling window of movements against
                       every template in the GestureLibrary; returns the
                       best match under a live-adjustable threshold
                       (`[`/`]` keys), subject to per-template cooldown

ActionDispatcher.dispatch(name)            (actions.py)
  looks the name up in the fixed action set; if mapped, fires the
  pyautogui call
```

This runs unconditionally on every frame with a detected hand, independently
of cursor mode below — gesture matching and cursor control read the same
per-frame hand detection but don't gate each other.

### Cursor mode (independent of the flow above)

```
CursorPipeline = HandCenterExtractor | ScreenPointConverter        (cursor/)
  -> HandCenterExtractor    Hand -> NormalizedPoint: mean (x, y) over all
                            21 image-space landmarks - deliberately not a
                            single fingertip, so the tracked point stays
                            stable while fingers move through a gesture;
                            this is what lets cursor mode and gesture
                            matching run at once off the same hand
  -> ScreenPointConverter   NormalizedPoint -> ScreenPoint: dead-zone hold
                            -> region-to-screen remap -> EMA smoothing;
                            a held hand yields None
  -> CursorDriver.move_to(point)             (cursor/driver.py)
                            the only part that touches pyautogui
```

Disabled, `CursorPipeline` yields nothing; re-enabling it resets the
position history rather than sliding over from where the hand was last
seen.

Toggled by `c` (configurable, `keybindings.toggle_cursor`) in
`apps/desktop.py`;
independent of the `r` recording toggle. See `docs/DESIGN_MATH.md` for why
the dead-zone + EMA combination is needed.

## Recording lifecycle

`r` toggles recording (driven from `DesktopApp._toggle_recording`, through
`GesturePipeline.start_recording()` / `stop_recording()`):
- **start**: clears the movement buffer and resets the running
  quantized-pose tracker, so the template opens with the pose the hand is
  in now rather than with its first change.
- **stop**: prompts (in the terminal, via `input()` — this blocks the camera
  loop, which is why `apps/desktop.py` draws a "check terminal" hint on the
  frame
  first) for an action name, e.g. `left-click`. The name is slugified into a
  filename (collisions get `-2`, `-3`, ... suffixes rather than overwriting).
  `GestureLibrary.add()` resolves the name, writes
  `recordings/<slug>.npz` (see below) and admits the template in memory, so
  the new gesture is immediately matchable without a reload.

## Recording file format

`recordings/*.npz` (binary, `np.savez_compressed`), not JSON — chosen for
size/parse speed over human-readability once the format stabilized:

| key          | dtype   | shape  | meaning                              |
|--------------|---------|--------|---------------------------------------|
| `bin_size`   | float32 | scalar | degrees per quantization bin          |
| `angle_bins` | int16   | (T,15) | quantized angle **bin index**, not degrees — reconstruct via `index * bin_size` |

`GestureLibrary` is the only reader and the only writer; it reconstructs
degrees on load. No timing is stored — see
`docs/adr/0001-recording-format-stores-no-timing.md`. Recordings written
before that change carry extra `t_ms`/`hysteresis` keys and still load,
because the reader never touched them.

## Module map

| module                          | role                                                            |
|----------------------------------|------------------------------------------------------------------|
| `stage.py`                      | `Stage`/`OptionalStage`/`Fork` — the composition primitives      |
| `domain.py`                     | the types that travel between stages                             |
| `vision/capture.py`             | `CaptureManager` — webcam -> stream of mirrored, timestamped frames |
| `vision/detection.py`           | `HandDetector` — MediaPipe wiring + thread-safe latest-result buffer |
| `vision/model_asset.py`         | `ensure_model()` — first-run model download                      |
| `gestures/angles.py`            | `AngleExtractor` — 21 landmarks -> 15 joint angles               |
| `gestures/movement.py`          | `MovementExtractor` — binning with hysteresis (Schmitt trigger), emitting only on change |
| `gestures/recorder.py`          | `GestureRecorder` — diverts movements out of the stream while recording |
| `gestures/library.py`           | `GestureLibrary` — the stored templates, name resolution, `.npz` I/O |
| `gestures/matcher.py`           | `GestureMatcher` — DTW live matching against the library         |
| `gestures/pipeline.py`          | `GesturePipeline` — the four above, composed; the recording API  |
| `cursor/center.py`              | `HandCenterExtractor` — hand -> normalized centre point          |
| `cursor/screen.py`              | `ScreenPointConverter` — dead zone + region remap + EMA smoothing |
| `cursor/pipeline.py`            | `CursorPipeline` — the two above, composed, with an on/off switch |
| `cursor/driver.py`              | `CursorDriver` — the only `pyautogui` cursor call                |
| `actions.py`                    | `ActionDispatcher` — gesture name -> `pyautogui` action          |
| `apps/overlay.py`               | frame drawing: hand skeleton + HUD text                          |
| `config.py`                     | `AppConfig` dataclasses + `load_config()` (reads `config.json`)  |
| `apps/desktop.py`               | `DesktopApp` — camera loop, key handling, ties it all together   |
| `apps/web.py`                   | the Streamlit demo (report-only, no `pyautogui`)                 |
| `__main__.py`                   | entry point (`python -m hand_recognition`)                       |
| `model/`                        | scaffolding for a hand-trained landmark model; see `docs/TRAINING.md` |

See `docs/DESIGN_MATH.md` for why each of these pieces works the way it does.
