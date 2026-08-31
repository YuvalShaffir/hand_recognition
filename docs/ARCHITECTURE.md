# Architecture

## Entry point

`src/hand_recognition/` is the real application, an installable package
(`pyproject.toml` declares it via `hatchling`, src-layout auto-detected from
the project name). `main.py` at the repo root is a **leftover bare-webcam
prototype** (opens a camera, shows the frame, no hand tracking or gesture
logic) — despite the name, it is not part of the pipeline, isn't part of
the package, and isn't kept in sync with the rest of the code.

Within the package:

- `__main__.py` — thin entry point (`App().run()`), reached via
  `python -m hand_recognition` or the `hand-recognition` console script.
- `app.py` — `App`: owns all live state (recorder, matcher, cursor,
  threshold) and runs the camera/detection loop and key handling. This is
  what used to be the monolithic `prot.py`.
- `landmarker.py` — MediaPipe `HandLandmarker` wiring (`create_landmarker`)
  and `HandLandmarkBuffer`, a thread-safe holder for the latest async
  detection result.
- `overlay.py` — frame drawing: hand skeleton (`draw_landmarks`) and HUD
  text (`draw_hud`, `draw_recording_prompt_hint`).
- `config.py` — `AppConfig` dataclass tree + `load_config()`; see
  "Configuration" below.

`gesture_recognizer.task` (repo root) is an unused leftover asset — no code
currently loads it. Only `hand_landmarker.task` is used, by `landmarker.py`.

Neither `.task` file is committed (`*.task` is gitignored — they're
generated binary assets, ~8MB each). `landmarker.ensure_model()` downloads
`hand_landmarker.task` from MediaPipe's official hosted URL on first run if
it's missing, so a fresh clone only needs `uv run hand-recognition` (plus
one-time network access) — no manual asset step.

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
`config.json`). `App()` calls `load_config()` itself when constructed
without an explicit `AppConfig`, then threads the relevant section into
each component's constructor (e.g. `CursorController(region_margin=...,
smoothing=..., deadzone=...)`).

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

```
camera frame
  -> HandLandmarker.detect_async()          (landmarker.py, via App.run())
  -> HandLandmarkBuffer.on_result() callback stores two views of the same
     hand, under a lock:
       latest_landmarks        - image-space (pixel drawing only, cursor mode)
       latest_world_landmarks  - real-world metric 3D (gesture features)
  -> hand_joint_angles(world_hand)           (hand_angles.py)
       21 landmarks -> 15 joint angles (degrees)
  -> GestureRecorder.observe(angles, now_ms) (recorder.py)
       quantizes via quantize_angles() (quantize.py, with hysteresis)
       returns True only when the quantized pose crosses a bin
       if recording: appends {t_ms, angles} on that change only
                     (event-driven -> run-length-collapsed sequence)
  -> when a bin changes AND not recording:
       GestureMatcher.observe(quantized, now_ms)   (gesture_dtw.py)
         DTW-aligns a rolling buffer of quantized events against every
         loaded template; returns the best match under a live-adjustable
         threshold (`[`/`]` keys), subject to per-template cooldown
  -> run_action(matched_name, actions)       (actions.py)
       looks up matched_name in the actions dict built by build_actions();
       if mapped, fires the pyautogui call
```

This runs unconditionally on every frame with a detected hand, independently
of cursor mode below — gesture matching and cursor control read the same
per-frame hand detection but don't gate each other.

### Cursor mode (independent of the flow above)

```
camera frame
  -> latest_landmarks (image-space, [0,1] normalized)  (landmarker.py, same detect_async() call)
  -> hand_centroid(hand)                     (cursor_control.py)
       mean (x, y) over all 21 landmarks - deliberately not a single
       fingertip, so the tracked point stays stable while fingers move
       through a gesture; this is what lets cursor mode and gesture
       matching run at the same time off the same hand
  -> CursorController.update(x, y)           (cursor_control.py)
       dead-zone hold -> region-to-screen remap -> EMA smoothing -> pyautogui.moveTo()
```

Toggled by `c` (configurable, `keybindings.toggle_cursor`) in `app.py`;
independent of the `r` recording toggle. See `docs/DESIGN_MATH.md` for why
the dead-zone + EMA combination is needed.

## Recording lifecycle

`r` toggles `GestureRecorder` (driven from `App._handle_record_toggle`):
- **start**: clears the frame buffer, resets the running quantized-pose
  tracker.
- **stop**: prompts (in the terminal, via `input()` — this blocks the camera
  loop, which is why `app.py` draws a "check terminal" hint on the frame
  first) for an action name, e.g. `left-click`. The name is slugified into a
  filename (collisions get `-2`, `-3`, ... suffixes rather than overwriting).
  Saved to `recordings/<slug>.npz` (see below), then templates are reloaded
  so the new recording is immediately matchable.

## Recording file format

`recordings/*.npz` (binary, `np.savez_compressed`), not JSON — chosen for
size/parse speed over human-readability once the format stabilized:

| key          | dtype   | shape  | meaning                              |
|--------------|---------|--------|---------------------------------------|
| `bin_size`   | float32 | scalar | degrees per quantization bin          |
| `hysteresis` | float32 | scalar | degrees of boundary margin            |
| `t_ms`       | int32   | (T,)   | ms since recording start, per frame   |
| `angle_bins` | int16   | (T,15) | quantized angle **bin index**, not degrees — reconstruct via `index * bin_size` |

`gesture_dtw.load_templates()` is the only reader; it reconstructs degrees
on load. `t_ms` is stored but not currently consumed by matching — see
`docs/DESIGN_MATH.md`.

## Module map

| module                          | role                                                            |
|----------------------------------|------------------------------------------------------------------|
| `hand_angles.py`                | 21 landmarks -> 15 joint-angle feature vector                    |
| `quantize.py`                   | per-angle binning with hysteresis (Schmitt trigger)               |
| `recorder.py`                   | `GestureRecorder` — continuous quantized-pose tracking + event-driven recording + `.npz` I/O |
| `gesture_dtw.py`                | `GestureTemplate`/`load_templates`, `GestureMatcher` (DTW live matching) |
| `cursor_control.py`             | `hand_centroid`, `CursorController` — hand position -> real OS cursor (dead-zone + EMA smoothing) |
| `actions.py`                    | `build_actions`/`run_action` — recording-name -> `pyautogui` action registry |
| `landmarker.py`                 | MediaPipe `HandLandmarker` wiring + thread-safe latest-result buffer + first-run model download |
| `overlay.py`                    | frame drawing: hand skeleton + HUD text                          |
| `config.py`                     | `AppConfig` dataclasses + `load_config()` (reads `config.json`)  |
| `app.py`                        | `App` — camera loop, key handling, ties it all together          |
| `__main__.py`                   | entry point (`python -m hand_recognition`)                       |
| `model/`                        | scaffolding for a hand-hand-trained landmark model; see `docs/TRAINING.md` |

See `docs/DESIGN_MATH.md` for why each of these pieces works the way it does.
