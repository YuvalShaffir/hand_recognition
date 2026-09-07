# Test Plan

The `pytest` suite for `hand_recognition`, in `tests/`. This document is the
specification it was written from, and the checklist for knowing when it is
done. It is **built**: run it with
`UV_PROJECT_ENVIRONMENT=.venv-wsl uv run pytest`. Where the suite as built
departs from what is written below, the last section says so and why.

Designed around one constraint: the Streamlit demo is publicly reachable, so
several strangers share one process. Everything below follows from that.

## Principles

- **A stage is the unit.** Test files mirror the source tree, one file per
  stage or collaborator. Two cross-cutting files (`test_pipeline_laws.py`,
  `test_import_purity.py`) hold the invariants that belong to no single
  stage.
- **No vision, no OS, no camera, no network.** The whole suite runs under
  WSL (`UV_PROJECT_ENVIRONMENT=.venv-wsl uv run pytest`) with no display, no
  webcam, and no `hand_landmarker.task` on disk. `mediapipe`, `cv2`,
  `pyautogui` and `urllib` are faked at their seams. No skip markers: a test
  that only runs somewhere else is a test that does not exist.
- **Fake at the boundary, never in the middle.** `pytest-mock` is for the
  four外 edges above. A test that patches a `hand_recognition` symbol to make
  another `hand_recognition` symbol pass is testing the mock.
- **Properties where the maths is.** `hypothesis` covers the numeric
  invariants (`joint_angles`, `quantize_angles`, `dtw_distance`); hand-written
  tables cover everything else.
- **Stress means bounds, not stopwatches.** Assert window sizes, cell counts
  and cross-instance isolation. Never assert wall-clock time — it measures
  the CI box, not the code.

## Prerequisites

The suite could not be written against the code as it stood. Three
production changes came first (all landed):

1. **`Landmark` protocol** (`domain.py`, issue #9). `Landmarks = Sequence[Any]` today,
   so every downstream stage duck-types objects only MediaPipe produces.
   Replace `Any` with a `Protocol` carrying `x: float`, `y: float`,
   `z: float`. Nothing changes at runtime; tests gain a typed seam and mypy
   checks it.
2. **Split persistence out of `GestureLibrary`** (`gestures/`, issues #6 and
   #7, blocked by #4).
   `GestureLibrary.__init__` globs a directory unconditionally, so a browser
   session inherits every template on the server's disk, and `add` writes
   into that same shared directory. Post-split: `GestureLibrary` is
   in-memory, holds no path, and is constructed empty. Loading and saving
   become two functions in `gestures/persistence.py` —
   `load_templates(directory) -> list[GestureTemplate]` and
   `save_template(directory, template) -> Path`. The desktop app seeds a
   library with the former and calls the latter after each `add`; the web app
   calls neither. Name uniqueness *within a library* stays the library's job;
   filename collision on disk becomes `save_template`'s.
3. **Test tooling** (issue #11). Add `pytest-cov` and `hypothesis` to `pyproject.toml`,
   plus a `[tool.pytest.ini_options]` section and the coverage gate below.

## Definition of done

Every named case in this document exists and passes, **and**:

```
--cov=src/hand_recognition --cov-fail-under=90
--cov-omit=src/hand_recognition/apps/*,src/hand_recognition/model/*
```

`apps/` is omitted because `desktop.py` is a `cv2` window loop around a
blocking `input()` and `web.py` is Streamlit-driven — both are reachable only
through mocking so heavy the test asserts nothing. Two pieces of pure logic
living inside `apps/` are tested anyway despite the omission: `_Keys` and
`DesktopApp._step_threshold`.

## Implementation order

The suite is built as twelve sequenced tasks, tracked as issues. Each is one
agent session: a coherent commit, independently verifiable, with the
prerequisites it needs already landed.

| # | Task | Blocked by |
|---|------|------------|
| #15 | Test scaffolding and the fixture factories | #9, #11 |
| #16 | Pipeline laws and import purity | #15 |
| #17 | Angle extraction and movement quantization | #15 |
| #18 | DTW distance and the gesture matcher | #15 |
| #19 | The gesture recorder and the in-memory library | #15, #6, #7 |
| #20 | `.npz` persistence, including hostile files | #19, #14 |
| #21 | The cursor stages | #15 |
| #22 | The vision stages | #15 |
| #23 | Config, actions, domain, desktop carve-outs | #15, #12 |
| #24 | The gesture and cursor pipelines end to end | #17, #18, #19, #21 |
| #25 | Session isolation and algorithmic bounds | #24, #13 |
| #26 | Turn on the coverage gate | #16, #20, #22, #23, #25 |

#15 is the only true bottleneck: once it lands, #16-#19 and #21-#23 can run
in any order, or at once. The gate in #26 goes on last — enabling it before
the suite exists just breaks the build.

## Layout

```
tests/
├── conftest.py                     # landmark/hand/pose/movement factories
├── test_pipeline_laws.py           # cross-cutting: stage.py invariants
├── test_import_purity.py           # cross-cutting: no pyautogui in the web path
├── test_isolation.py               # cross-cutting: N-session stress
├── test_config.py
├── test_actions.py
├── test_domain.py
├── gestures/
│   ├── test_angles.py
│   ├── test_movement.py
│   ├── test_recorder.py
│   ├── test_library.py
│   ├── test_persistence.py
│   ├── test_matcher.py
│   └── test_pipeline.py
├── cursor/
│   ├── test_center.py
│   ├── test_screen.py
│   ├── test_pipeline.py
│   └── test_driver.py
├── vision/
│   ├── test_capture.py
│   ├── test_detection.py
│   └── test_model_asset.py
└── apps/
    └── test_desktop_logic.py       # _Keys and _step_threshold only
```

### `conftest.py`

The whole suite's dependency budget lives here.

- `landmark(x, y, z)` — a `SimpleNamespace` satisfying the `Landmark`
  protocol.
- `flat_hand()` — 21 landmarks in a known, fully-extended configuration whose
  joint angles are all 180°, by construction.
- `fist_hand()` — 21 landmarks whose joint angles are all ~0°.
- `hand(landmarks=..., world_landmarks=..., timestamp_ms=...)` — a `Hand`
  with sensible defaults, so a test names only what it cares about.
- `pose(angles, timestamp_ms)`, `movement(angles, timestamp_ms)`.
- `template(name, frames, bin_size)` — a `GestureTemplate` from a plain list.
- `landmark_arrays()` — a `hypothesis` strategy: 21 finite 3-vectors in a
  plausible coordinate range.

---

# Cross-cutting

## `test_pipeline_laws.py`

The two rules `CLAUDE.md` calls load-bearing are invisible to per-stage
tests: every unit test still passes when a stage skips instead of returning
`None`, while the `Fork` branches silently desynchronise. Fake stages only —
no domain types needed.

**Happy**
- `test_chain_applies_left_to_right` — `A | B` calls `A` then `B`.
- `test_chain_is_itself_a_stage` — `(A | B) | C` and `A | (B | C)` produce
  the same output for the same input.
- `test_call_maps_a_stream_one_to_one` — a stage called on an iterator of *n*
  items yields exactly *n* items.
- `test_fork_pairs_both_branch_results` — `Fork(A, B).apply(x) == (A(x), B(x))`.

**The `None` law**
- `test_optional_stage_passes_none_through_untransformed` — `transform` is
  never called for `None`.
- `test_optional_stage_returning_none_still_yields_an_item` — a stream of *n*
  items through a stage that reports nothing yields *n* `None`s, not zero
  items. **This is the one that catches a skipping stage.**
- `test_chain_of_optional_stages_preserves_stream_length` — three stacked
  optional stages, each reporting nothing half the time, still yield *n*.
- `test_fork_branches_stay_in_lockstep` — drive `Fork` with *n* items where
  each branch reports nothing on a different subset; assert *n* pairs out and
  that pair *i* corresponds to input *i*.

**Real pipelines**
- `test_gesture_pipeline_yields_one_result_per_hand` — including `None` hands.
- `test_cursor_pipeline_yields_one_result_per_hand` — enabled and disabled.

## `test_import_purity.py`

`web.py`'s docstring claims it must not import `actions.py` or
`cursor.driver`, because both touch `pyautogui` at module scope
(`pyautogui.FAILSAFE = True`) and that fails outright on a headless
container. Nothing checks it today.

- `test_recognition_modules_import_without_pyautogui` — with `pyautogui`
  forced to raise on import (a `sys.modules` sentinel + a fresh import in a
  subprocess), import `domain`, `stage`, `config`, every module under
  `gestures/`, `vision/`, and `cursor/` *except* `cursor.driver`. All succeed.
- `test_web_app_module_graph_excludes_pyautogui` — walk `apps.web`'s
  transitive imports statically (`ast`, not execution — importing Streamlit is
  not this test's business) and assert neither `hand_recognition.actions` nor
  `hand_recognition.cursor.driver` appears.
- `test_cursor_package_init_does_not_pull_the_driver` — `cursor/__init__.py`
  exports `CursorPipeline`; assert importing the package does not import
  `cursor.driver`.

## `test_isolation.py` — stress

The invariant bought by making libraries session-scoped: N sessions in one
process do not see each other. Pure stages given their inputs, so this stays
deterministic — the threads probe locking and shared state, not timing.

- `test_parallel_pipelines_do_not_share_templates` — 16 `GesturePipeline`s in
  16 threads, each recording a differently-named gesture; afterwards each
  library holds exactly its own one template.
- `test_parallel_pipelines_do_not_share_movement_state` — drive 16 pipelines
  with 16 different movement streams concurrently; each pipeline's output
  matches what it produces when run alone.
- `test_matcher_cooldown_is_per_instance` — a match in one matcher never
  cools down another.
- `test_latest_hands_is_safe_under_concurrent_writes` — hammer
  `_LatestHands.on_result` from several threads while reading `get()`;
  every value read is one complete tuple that was written, never a mixture.
- `test_no_module_level_mutable_state` — construct two of every stateful
  stage, mutate the first, assert the second is untouched.

**Bounds (no wall-clock assertions)**
- `test_matcher_window_never_exceeds_twice_the_longest_template` — feed 10,000
  movements; assert `len(matcher._window) <= 2 * library.longest` throughout.
- `test_dtw_cell_count_stays_within_budget` — instrument the step-matrix size
  and assert it is `O(n·m)` with the expected constants for known input sizes.
- `test_recorder_memory_is_bounded_by_the_cap` — see security item 4 below;
  record 10,000 movements and assert the collected count stops at the cap.
- `test_matcher_cost_is_bounded_by_the_template_length_cap` — with the cap in
  place, a hostile 10,000-frame template cannot be admitted, so per-frame cost
  has a ceiling.

---

# `stage.py` / `domain.py` / `config.py` / `actions.py`

## `test_domain.py`

- `test_detection_primary_returns_the_first_hand` (happy).
- `test_detection_primary_is_none_with_no_hands` (sad).
- `test_landmark_protocol_accepts_a_plain_namespace` — the seam the whole
  suite rests on; a static `assert_type`-style check plus a runtime use.
- `test_frozen_dataclasses_reject_mutation` — `Frame`, `Hand`, `Pose`,
  `Movement` are frozen; `GestureTemplate` deliberately is not (`add`
  rewrites `name`). Documents the asymmetry so it is not "fixed" by accident.

## `test_config.py`

`_section` rejects unknown *keys* but validates no *values*. That is a
security finding (item 3), not just a gap.

**Happy**
- `test_missing_file_yields_all_defaults` — `load_config("nope.json") == AppConfig()`.
- `test_empty_object_yields_all_defaults`.
- `test_partial_section_keeps_sibling_defaults` — `{"camera": {"width": 1280}}`
  leaves `height` at 480.
- `test_every_section_is_overridable` — parametrised across all eight sections.

**Sad**
- `test_unknown_section_raises` — message names the offending section.
- `test_unknown_key_raises` — message names the offending key.
- `test_malformed_json_raises_json_decode_error`.
- `test_directory_passed_as_path_raises` — currently an `IsADirectoryError`
  from `read_text`; assert the real behaviour, note it.

**Edge**
- `test_accepts_str_and_path` — both forms of the `path` argument.
- `test_null_value_for_a_key` — `{"camera": {"width": null}}`.

**Security — value validation (see finding 3)**
These are written *against the fix*, and fail until it lands:
- `test_zero_bin_size_is_rejected` — `bin_size_deg: 0` currently reaches
  `quantize_angles` and divides by zero.
- `test_negative_bin_size_is_rejected`.
- `test_negative_hysteresis_is_rejected`.
- `test_threshold_min_above_max_is_rejected` — makes `_step_threshold`'s
  clamp incoherent.
- `test_absurd_num_hands_is_rejected` — `num_hands: 100000`.
- `test_negative_deadzone_is_rejected`, `test_smoothing_outside_0_1_is_rejected`.
- `test_wrong_type_is_rejected` — `{"camera": {"width": "wide"}}` currently
  constructs a `CameraConfig` with a `str` and fails much later, inside `cv2`.
- `test_camera_dimensions_must_be_positive`.

## `test_actions.py`

`pyautogui` faked at the module seam. Assert *which* call is made, not that a
click happened.

**Happy**
- `test_dispatch_left_click_calls_pyautogui_click_left` (and right, double).
- `test_scroll_up_uses_the_configured_amount` — `scroll_amount` from config.
- `test_scroll_down_negates_the_amount`.
- `test_dispatch_returns_true_when_an_action_fired`.
- `test_names_lists_the_five_known_actions`.

**Sad**
- `test_unknown_name_returns_false_and_fires_nothing` — the "matched but not a
  macro" case from `CONTEXT.md`.
- `test_empty_name_returns_false`.
- `test_dispatch_propagates_a_pyautogui_failsafe` — `FailSafeException` is a
  deliberate user abort and must not be swallowed.

**Edge**
- `test_names_are_case_sensitive` — `"Left-Click"` is not `"left-click"`.
- `test_zero_scroll_amount_still_calls_scroll`.
- `test_failsafe_is_enabled_on_import` — `pyautogui.FAILSAFE is True`.

---

# `gestures/`

## `test_angles.py`

**Happy**
- `test_flat_hand_gives_straight_angles` — all 15 ≈ 180°.
- `test_fist_hand_gives_folded_angles` — all 15 ≈ 0°.
- `test_returns_fifteen_angles` — `NUM_ANGLES == 15`, three per finger.
- `test_triples_match_the_finger_chains` — the precomputed `_TRIPLES` really
  are the consecutive triples of `FINGER_CHAINS`, so the optimisation cannot
  drift from the definition it replaced.
- `test_extractor_uses_world_landmarks_not_image_landmarks` — a hand whose two
  landmark sets differ; the pose must come from the world set. The point of
  `docs/DESIGN_MATH.md`, and invisible in normal use.
- `test_extractor_preserves_the_timestamp`.

**Edge / properties (`hypothesis`)**
- `test_angles_are_always_within_zero_and_one_eighty` — over
  `landmark_arrays()`, no `nan` escapes `arccos`.
- `test_angles_are_invariant_under_translation` — the invariance the design
  claims.
- `test_angles_are_invariant_under_rotation` — apply a random rotation matrix.
- `test_angles_are_invariant_under_uniform_scale`.
- `test_collinear_points_do_not_produce_nan` — `cos` of exactly ±1; the
  `np.clip` guard.
- `test_coincident_points_produce_nan_not_a_crash` — a zero-length vector
  divides by zero. **Document the current behaviour and decide**: MediaPipe
  should never emit duplicate landmarks, but a malformed `.npz` or a future
  local model could. Candidate for a `nan`-to-zero guard.

**Sad**
- `test_too_few_landmarks_raises` — 20 landmarks; currently an `IndexError`.
- `test_empty_landmarks_raises`.
- `test_extractor_passes_none_through` — the `OptionalStage` law.

## `test_movement.py`

**Happy**
- `test_first_pose_always_emits_a_movement` — no previous state to compare to.
- `test_unchanged_pose_emits_nothing` — a hand held still produces no
  movements, however many frames pass.
- `test_changed_pose_emits_a_movement`.
- `test_angles_snap_to_bin_centres` — multiples of `bin_size`.
- `test_movement_carries_the_pose_timestamp`.

**Hysteresis (the reason this module exists)**
- `test_jitter_inside_the_margin_does_not_change_the_bin` — oscillate an
  angle across a bin edge by less than `bin_size/2 + hysteresis`; assert one
  movement total, not one per frame.
- `test_crossing_the_margin_commits_the_new_bin`.
- `test_hysteresis_is_per_joint` — one joint moving does not drag the other
  fourteen along.
- `test_hysteresis_property` (`hypothesis`) — for any angle sequence whose
  successive deltas stay under the margin, at most one movement is emitted
  after the first.

**Edge**
- `test_reset_clears_the_previous_pose` — the next pose emits again, which is
  what makes `start_recording` open with the hand's current pose.
- `test_zero_bin_size_divides_by_zero` — pins the behaviour the config fix
  makes unreachable.
- `test_zero_hysteresis_still_quantizes` — hysteresis is optional, bins are not.
- `test_hysteresis_wider_than_the_bin_freezes_the_pose` — a legal-but-useless
  configuration; assert it degrades rather than misbehaves.
- `test_nan_angles_never_equal_themselves` — `np.array_equal` with `nan`
  makes every frame look like a movement. Interacts with the `arccos` edge
  case above; worth knowing before it is met in production.

**Sad**
- `test_passes_none_through`.
- `test_wrong_angle_count_is_handled` — a 14-element pose.

## `test_recorder.py`

**Happy**
- `test_passes_movements_through_when_not_recording`.
- `test_swallows_movements_while_recording` — returns `None`, so a gesture
  being recorded cannot fire a macro as it is performed.
- `test_collects_movements_while_recording` — `frame_count` tracks them.
- `test_finish_builds_a_template_of_the_collected_movements` — shape
  `(T, 15)`, `bin_size` carried through.
- `test_finish_stops_recording`.
- `test_start_discards_a_previous_partial_recording`.

**Sad / edge**
- `test_finish_without_recording_returns_an_empty_template` — shape `(0,)`;
  the case `GesturePipeline.stop_recording` turns into `None`.
- `test_finish_twice_returns_an_empty_second_template`.
- `test_single_movement_recording` — a one-frame template, admissible and
  degenerate for DTW.
- `test_passes_none_through_while_recording` — the `None` law holds even in
  the diverted state.
- `test_empty_name_is_accepted` — naming is the library's problem.

**Security — item 4**
- `test_recording_stops_collecting_at_the_cap` — written against the fix. A
  browser session that starts recording and walks away currently grows a list
  forever; on a shared container that is one tab consuming the box.

## `test_library.py`

Post-split: in-memory, holds no path, touches no disk. **Every test in this
file runs with no filesystem at all** — if one needs `tmp_path`, the split in
prerequisite 2 is incomplete.

**Happy**
- `test_a_new_library_is_empty` — the property that closes the leak: a fresh
  library inherits nothing.
- `test_add_returns_the_stored_name`.
- `test_add_appends_to_templates`.
- `test_names_lists_stored_templates_in_insertion_order`.
- `test_longest_reports_the_longest_template_length`.
- `test_seeding_from_loaded_templates` — the desktop path: `load_templates`
  output handed in at construction.

**Naming**
- `test_name_is_slugified` — `"Left Click"` → `"left-click"`.
- `test_collision_appends_a_counter` — second `"wave"` becomes `"wave-2"`.
- `test_repeated_collisions_increment` — `wave-3`, `wave-4`.
- `test_blank_name_falls_back_to_a_timestamp` — `gesture_YYYYMMDD_HHMMSS`;
  freeze the clock.
- `test_name_of_only_punctuation_falls_back_to_a_timestamp` — `"!!!"`
  slugifies to empty.
- `test_unicode_name_falls_back_to_a_timestamp` — `"✋"` has no `[a-z0-9]`.
  Worth deciding: silently timestamping a name the user typed is surprising.

**Sad**
- `test_add_rejects_a_template_with_no_frames` — `ValueError`.
- `test_longest_defaults_to_one_when_empty` — the divide-by-zero guard the
  matcher's window sizing depends on.
- `test_templates_property_is_a_tuple` — callers cannot mutate the library's
  list through it.

**Security — item 1**
- `test_traversal_in_a_name_is_neutralised` — `"../../etc/passwd"` →
  `"etc-passwd"`. `_slugify` already handles this; the test stops a future
  "friendlier names" change from reopening it.
- `test_dot_dot_falls_back_to_a_timestamp` — `".."` slugifies to empty.
- `test_windows_reserved_names_are_not_special_cased` — `"CON"` → `"con"`,
  which is still a reserved device name on Windows. Currently unhandled;
  belongs to `save_template`, tested there.
- `test_very_long_name_is_truncated` — 5,000 characters. Written against the
  fix; see `test_persistence.py`.

## `test_persistence.py`

New module (prerequisite 2). The *only* file in the suite that touches the
filesystem, via `tmp_path`. All `.npz` hostility concentrates here.

**Happy**
- `test_save_then_load_round_trips_a_template` — name, `bin_size` and frames
  survive.
- `test_load_returns_templates_sorted_by_name` — deterministic ordering.
- `test_save_creates_the_directory_if_absent`.
- `test_load_from_a_missing_directory_returns_empty` — not an exception; a
  first run has no `recordings/`.
- `test_load_ignores_non_npz_files`.
- `test_quantization_survives_the_int16_round_trip` — frames are stored as
  bin *indices* and multiplied back by `bin_size` on load; assert the
  reconstructed angles match within one ulp of a bin.

**Sad**
- `test_empty_template_file_is_skipped` — zero frames, currently skipped
  silently by `_load`. Assert it stays silent-but-safe, and log.
- `test_unreadable_directory_raises` — permissions.
- `test_save_to_a_full_or_readonly_directory_raises_oserror`.

**Security — item 2** (written against the fix; `.npz` is attacker-supplied
whenever a deployment enables persistence)
- `test_missing_bin_size_key_is_rejected` — currently an uncaught `KeyError`
  at library construction, so one bad file bricks startup.
- `test_missing_angle_bins_key_is_rejected`.
- `test_wrong_angle_width_is_rejected` — `(T, 7)` instead of `(T, 15)`
  currently reaches the matcher and broadcasts wrong or explodes mid-frame.
- `test_one_dimensional_angle_bins_is_rejected`.
- `test_non_numeric_dtype_is_rejected`.
- `test_zero_or_negative_bin_size_is_rejected` — a stored `bin_size` of 0
  divides by zero inside `dtw_distance`.
- `test_nan_or_inf_in_frames_is_rejected` — poisons every DTW comparison.
- `test_oversized_array_is_rejected` — a frame-count ceiling; the same cap as
  security item 4, enforced on load.
- `test_decompression_bomb_is_rejected` — a few-KB `.npz` inflating to
  gigabytes. `np.load` will do this happily; check the declared shape before
  materialising the array.
- `test_pickle_payload_is_refused` — `allow_pickle` defaults to `False`, so
  this already fails; the test pins the default against a future
  `allow_pickle=True` convenience change. **The one place a bad file is
  RCE rather than a crash.**
- `test_very_long_filename_raises_oserror` — a 5,000-character name exceeds
  the 255-byte limit; currently an uncaught `OSError` from `savez_compressed`.
  Truncate in `save_template`.
- `test_windows_reserved_filename_is_avoided` — `con.npz` is unopenable on
  Windows, which is the platform the desktop app targets.

## `test_matcher.py`

### `dtw_distance`

**Happy**
- `test_identical_sequences_cost_zero`.
- `test_cost_is_in_bin_units` — a one-bin offset across all joints costs 1.
- `test_time_warping_is_free` — a sequence and its 2× time-stretch cost 0,
  the property that lets the same gesture be performed fast or slow.
- `test_cost_is_symmetric` (`hypothesis`).
- `test_cost_is_non_negative` (`hypothesis`).
- `test_cost_grows_with_divergence` — monotonic in the offset size.

**Early abandonment** — the highest-value tests in the file: an optimisation
that returns a *different answer* is a silent recognition bug.
- `test_abandonment_matches_the_full_computation` (`hypothesis`) — for random
  pairs, `dtw_distance(a, b, s, abandon_above=t)` is either the exact
  unabandoned cost or `inf`, and it is `inf` only when the true cost ≥ `t`.
- `test_abandonment_never_discards_a_winner` — a pair whose true cost is just
  under the limit is still returned.
- `test_infinite_limit_never_abandons`.
- `test_zero_limit_abandons_immediately`.

**Edge / sad**
- `test_single_frame_sequences`.
- `test_length_one_against_length_many`.
- `test_empty_sequence_behaviour` — pin it; `n + m == 0` divides by zero.
- `test_mismatched_widths_raise` — 15 vs 7 columns.
- `test_zero_bin_size_divides_by_zero` — pinning what the `.npz` and config
  validation make unreachable.
- `test_nan_in_a_sequence_propagates` — documents why `nan` is rejected at
  the boundary rather than handled here.

### `GestureMatcher`

**Happy**
- `test_matches_a_template_performed_exactly`.
- `test_matches_a_template_performed_at_a_different_speed`.
- `test_returns_none_before_enough_movements_accumulate`.
- `test_returns_none_when_nothing_is_close_enough`.
- `test_picks_the_closest_of_several_templates`.
- `test_an_empty_library_never_matches`.

**Threshold**
- `test_tighter_threshold_rejects_a_loose_performance`.
- `test_looser_threshold_accepts_it`.
- `test_threshold_is_mutable_mid_stream` — the live `[`/`]` behaviour.

**Cooldown**
- `test_a_match_suppresses_the_same_gesture_within_the_cooldown` — the
  anti-double-fire guard.
- `test_the_same_gesture_matches_again_after_the_cooldown`.
- `test_cooldown_is_per_template` — gesture B still matches during A's
  cooldown.
- `test_zero_cooldown_allows_consecutive_matches`.
- `test_a_backwards_timestamp_does_not_permanently_suppress` — clock skew:
  `now_ms - last` goes negative and stays under the cooldown forever. A real
  hazard given `CaptureManager` and `web.py` derive timestamps independently
  from `time.monotonic()`.

**Window**
- `test_window_is_trimmed_to_twice_the_longest_template`.
- `test_window_survives_a_template_being_added_mid_stream` — recording ends
  and the library grows while the matcher is running.

**Sad**
- `test_passes_none_through`.
- `test_a_template_longer_than_the_window_never_matches` — a template longer
  than `2 × longest` cannot happen by construction; assert `longest` really
  is the bound.

## `test_pipeline.py` (gestures)

Wiring, not maths — each stage is tested above.

**Happy**
- `test_hand_to_gesture_name_end_to_end` — real stages, fake hands, no vision.
- `test_recording_diverts_movements_from_the_matcher` — a gesture cannot fire
  while it is being recorded.
- `test_stop_recording_admits_the_template_and_returns_its_name`.
- `test_the_recorded_gesture_matches_when_performed_again` — the whole point
  of the product, in one test.
- `test_start_recording_resets_the_movement_extractor` — the template opens
  with the hand's current pose, not its first change.
- `test_threshold_property_reaches_the_matcher`.
- `test_recorded_frame_count_reflects_the_recorder`.

**Sad**
- `test_stop_recording_returns_none_when_the_hand_never_moved`.
- `test_stop_recording_without_starting_returns_none`.
- `test_none_hands_yield_none`.
- `test_persist_false_leaves_the_filesystem_untouched` — the web path.

## `test_center.py`

- `test_centre_is_the_mean_of_all_landmarks` (happy).
- `test_centre_is_stable_while_fingers_move` — the design claim: a hand
  translating not at all but flexing its fingers barely moves its centre,
  which is what lets cursor mode and gesture matching share one hand.
- `test_uses_image_landmarks_not_world_landmarks` — the mirror image of the
  angles test, and equally invisible in normal use.
- `test_passes_none_through` (sad).
- `test_empty_landmarks_divides_by_zero` (edge) — pin it.
- `test_single_landmark_is_its_own_centre` (edge).

## `test_screen.py`

**Happy**
- `test_centre_of_the_active_region_maps_to_the_centre_of_the_screen`.
- `test_region_edges_map_to_screen_edges` — `region_margin` inset.
- `test_outside_the_region_clamps_to_the_screen_edge` — both axes, both ends.
- `test_output_scales_by_the_screen_size`.

**Dead zone and smoothing**
- `test_first_point_is_emitted_unsmoothed` — nothing to smooth against.
- `test_a_move_inside_the_deadzone_emits_nothing`.
- `test_a_move_past_the_deadzone_emits`.
- `test_deadzone_is_measured_against_the_committed_point_not_the_smoothed_one` —
  the two are deliberately different; a test that conflates them passes today
  and breaks on any refactor.
- `test_smoothing_moves_a_fraction_of_the_way` — exact EMA arithmetic.
- `test_repeated_identical_points_converge` (`hypothesis`) — feeding the same
  point converges monotonically toward it and never overshoots.
- `test_reset_clears_both_histories` — re-entering cursor mode does not snap
  or interpolate from a stale position.

**Edge**
- `test_zero_smoothing_freezes_the_cursor` — legal config, degenerate result.
- `test_smoothing_of_one_disables_smoothing`.
- `test_zero_deadzone_emits_every_frame`.
- `test_region_margin_of_zero_maps_the_whole_frame`.
- `test_region_margin_of_half_collapses_the_span` — divides by zero. Config
  validation (item 3) should forbid `>= 0.5`.
- `test_passes_none_through` (sad).

## `test_pipeline.py` (cursor)

- `test_disabled_pipeline_yields_none` (happy).
- `test_enabled_pipeline_yields_screen_points`.
- `test_enabling_resets_the_converter` — re-entry does not slide the cursor in
  from where the hand was last seen.
- `test_disabling_and_re-enabling_starts_clean`.
- `test_enabling_an_already_enabled_pipeline_does_not_reset` — the guard in
  the setter; a spurious reset would jump the cursor.
- `test_none_hands_yield_none_when_enabled` (sad).
- `test_stream_length_is_preserved_in_both_states` (the `None` law).

## `test_driver.py`

`pyautogui` faked. The only cursor module that touches it.

- `test_screen_size_is_read_once_at_construction` (happy).
- `test_move_to_calls_pyautogui_moveto_with_the_point`.
- `test_move_to_passes_pause_false` — `_pause=True` would throttle to
  ~10 Hz and make cursor mode unusable; a non-obvious argument worth pinning.
- `test_failsafe_is_enabled_on_import`.
- `test_a_failsafe_exception_propagates` (sad).

---

# `vision/`

## `test_capture.py`

`cv2.VideoCapture` faked; no webcam.

**Happy**
- `test_iterating_yields_frames_from_the_capture`.
- `test_frames_are_mirrored_by_default` — the user's hand moves with them.
- `test_mirroring_can_be_disabled`.
- `test_camera_config_is_applied` — index, width, height reach `cv2`.
- `test_context_manager_releases_the_capture`.

**Timestamps** — MediaPipe's live-stream mode *requires* strict monotonicity;
violating it is a hard error deep inside the graph.
- `test_timestamps_strictly_increase` over many frames.
- `test_timestamps_increase_even_when_the_clock_does_not` — two frames within
  the same millisecond still get distinct, increasing values. The reason
  `_next_timestamp_ms` has a `max(...)` in it.
- `test_timestamps_survive_a_backwards_clock` — property-based.

**Sad**
- `test_iterating_before_entering_raises_runtimeerror`.
- `test_a_failed_read_ends_the_stream` — unplugged webcam; stops cleanly.
- `test_a_closed_capture_yields_nothing`.
- `test_exit_is_safe_when_never_entered`.
- `test_exit_twice_is_safe`.

## `test_detection.py`

`mediapipe` faked at the `HandLandmarker` seam; no model file, no download.

**`_LatestHands`**
- `test_starts_empty` (happy).
- `test_on_result_pairs_landmarks_with_world_landmarks` — pairing them wrongly
  swaps the two coordinate spaces and breaks angles *and* the cursor at once.
- `test_on_result_stamps_the_callback_timestamp`.
- `test_a_result_with_no_hands_clears_the_previous_one` (sad).
- `test_get_returns_a_tuple_not_a_live_reference`.
- (Concurrency covered in `test_isolation.py`.)

**`HandDetector`**
- `test_apply_before_entering_raises_runtimeerror` (sad).
- `test_apply_submits_the_frame_asynchronously`.
- `test_apply_converts_bgr_to_rgb` — MediaPipe expects RGB; OpenCV gives BGR.
  Getting this wrong degrades detection quietly rather than failing.
- `test_apply_returns_the_latest_completed_result_not_this_frames` — the
  documented asynchrony; a test that assumes otherwise passes by luck.
- `test_landmarker_config_reaches_the_options`.
- `test_enter_ensures_the_model_is_present`.
- `test_exit_closes_the_landmarker`.
- `test_detection_carries_the_original_frame`.

## `test_model_asset.py`

`urllib.request.urlretrieve` faked; no network.

- `test_existing_file_is_not_downloaded` (happy).
- `test_missing_file_triggers_a_download_to_the_given_path`.
- `test_a_url_error_raises_runtimeerror_with_manual_instructions` (sad) — the
  message names the URL and the path, because this is the one failure a
  fresh clone hits offline.
- `test_a_partial_download_leaves_no_valid_looking_file` — `urlretrieve`
  writes as it goes, so an interrupted download leaves a truncated file that
  `path.exists()` accepts forever after. **A real bug**: download to a temp
  path and rename. Written against the fix.
- `test_the_model_url_is_https` — a plain-HTTP model fetch is an executable
  artefact over an unauthenticated channel.

---

# `apps/`

## `test_desktop_logic.py`

Omitted from coverage, tested anyway — pure logic that happens to live here.

**`_Keys`**
- `test_keybindings_map_to_ordinals`.
- `test_a_multi_character_binding_raises` — `ord("esc")` is a `TypeError`;
  config validation should catch it first.
- `test_an_empty_binding_raises`.

**`_step_threshold`**
- `test_tighten_decreases_by_the_step`.
- `test_loosen_increases_by_the_step`.
- `test_clamps_at_the_minimum` and `test_clamps_at_the_maximum`.
- `test_rounds_to_two_decimals` — repeated steps do not accumulate float
  drift into the HUD readout.

---

## Findings this plan documents rather than ratifies

Tests marked "written against the fix" fail until the corresponding issue
lands. They are written first deliberately: a test asserting that a `KeyError`
escapes is a test that blesses a crash.

| # | Finding | Reachable from a browser? |
|---|---------|---------------------------|
| 1 | `GestureLibrary` globs and writes a shared directory | **Yes** — #6, #7 |
| 2 | `config.json` values are never validated | **Yes** — zero `bin_size_deg` crashes the pipeline (#12) |
| 3 | Recorder and template lengths are unbounded | **Yes** — one tab can tax the container (#13) |
| 4 | `.npz` files are loaded without validation | No — desktop-local unless persistence is deployed (#14) |
| 5 | Long or reserved gesture names raise `OSError` | No — desktop-local (#14) |
| 6 | An interrupted model download is cached as valid | No — but it bricks a fresh clone permanently (unfiled) |

## Deviations from this plan, as built

The plan is the specification; where the code disagreed with it, these are
the places the code won.

- **`tests/` is a package.** `tests/gestures/test_pipeline.py` and
  `tests/cursor/test_pipeline.py` share a basename, which pytest cannot
  import without either `__init__.py` files or a non-default import mode.
  The suite has the `__init__.py` files, and shared factories import as
  `from ..conftest import ...`.
- **`pyautogui` is faked in `sys.modules`, not patched.** It raises on
  import on a headless machine, so there is nothing to patch afterwards;
  `conftest.py` installs a stand-in before any test module imports
  `actions.py` or `cursor/driver.py`. Still the boundary, just reached
  earlier.
- **`dtw_distance` normalises by `n + m`**, so a one-bin offset between two
  equal-length sequences costs 0.5, not 1. `test_cost_is_in_bin_units`
  asserts the invariance that actually holds: the cost depends on the offset
  in bins, so scaling the bin size and the offset together leaves it
  unchanged.
- **A gesture is recognised *within* its performance, not on its last
  movement.** The matcher's window slides, so the frame it fires on is not
  fixed; the `GestureMatcher` tests assert the name is fired somewhere in
  the performance.
- **`nan` in a sequence yields `inf`, not `nan`** — the abandonment
  comparison catches it first. Either way the cost is unusable, which is the
  point.
- **Angle invariances hold to 0.1°**, not to machine precision: `arccos` is
  ill-conditioned near 0° and 180°, the angles a real hand spends most of
  its time at.
- **`GestureRecorder.finish` now clears its buffer**, which is what
  `test_finish_twice_returns_an_empty_second_template` asks for and what
  frees a long recording's memory.
- **Renamed cases**, where the plan's name contradicted its own description
  or the fix it was written against: `test_very_long_filename_raises_oserror`
  → `test_very_long_filename_is_truncated`;
  `test_persist_false_leaves_the_filesystem_untouched` →
  `test_recording_leaves_the_filesystem_untouched` (the split removed the
  `persist` flag); `test_a_multi_character_binding_raises` and
  `test_an_empty_binding_raises` merged into one parametrised case.
- **An invalid recording is skipped and logged, not raised.** The plan's
  `.npz` cases are named `..._is_rejected`, and rejection that takes the
  whole library down with it just moves the "one bad file bricks startup"
  finding rather than fixing it. `assert_rejected` in
  `test_persistence.py` asserts the file is skipped *and* that the reason
  names it.
- **Findings 1-6 are fixed, not pinned.** Config values are validated on
  load, recorder and template lengths are capped
  (`domain.MAX_TEMPLATE_FRAMES`), `.npz` files are validated from the header
  before being read, long and Windows-reserved filenames are neutralised in
  `save_template`, the model downloads to a `.part` file and is renamed, and
  a backwards timestamp expires the match cooldown instead of freezing it.
  The tests written "against the fix" therefore pass.

## Not yet covered

`src/hand_recognition/model/` is empty. `docs/TRAINING.md` plans a
hand-written landmark regressor to replace MediaPipe, trained on FreiHAND. Its
tests are a separate plan, written when there is code to test: training-loop
testing is a different discipline (seeded determinism, tensor shape contracts,
overfit-a-single-batch smoke tests, checkpoint round-trips) and does not
belong in the same document as stage unit tests.
