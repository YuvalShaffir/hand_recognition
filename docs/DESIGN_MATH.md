# Design Math

Why the gesture pipeline is built the way it is — the reasoning, not just the
code. See `docs/ARCHITECTURE.md` for how these pieces connect.

## Feature vector: joint angles, not raw coordinates

`gestures/angles.py` reduces 21 landmarks to 15 numbers: the angle at each
finger joint, 3 per finger (`FINGER_CHAINS` walks wrist -> ... -> fingertip;
each consecutive triple `(a, b, c)` gives the angle at `b`):

```
angle(a, b, c) = degrees(arccos( (a-b)·(c-b) / (|a-b| |c-b|) ))
```

This is deliberately **not** raw x/y/z: a dot-product-based angle is
inherently translation-invariant (only vector *differences* matter) and
scale-invariant (normalized by the vector norms), and — the property the
recorder actually needs — rotation-invariant *provided the input coordinates
are a genuine rigid 3D representation*. A recorded gesture and a
live one performed with the hand at a different position, size-on-screen, or
orientation should produce the same angle vector.

### Why `hand_world_landmarks`, not `hand_landmarks`

MediaPipe's `hand_landmarks` are image-normalized: x/y are perspective-
projected into `[0,1]` relative to the frame, and z is a depth *proxy* on a
different noise/scale profile than x/y. That's not a rigid coordinate space
— rotating the physical hand does not correspond to a rigid rotation of
`(x,y,z)` in that representation, so angle values computed from it drift as
the hand turns toward/away from the camera, even though the math above is
"rotation-invariant" in principle.

`hand_world_landmarks` are MediaPipe's metric 3D reconstruction (meters,
de-projected) — a much closer approximation of true rigid hand geometry.
Feeding *those* into the same angle formula is what actually makes the
angle-invariance argument hold in practice. This is why `vision/detection.py` keeps
both views on every `Hand`: `landmarks` for pixel-space drawing and cursor
mode, `world_landmarks` for every gesture feature.

## Quantization + hysteresis

`gestures/movement.py` snaps each of the 15 angles to the nearest multiple of
`bin_size` (default 15°). A naive nearest-bin snap flickers when a raw angle
sits near a boundary (landmark jitter flips it back and forth every frame).
`quantize_angles()` fixes this with a per-dimension Schmitt trigger: the
current bin is held until the raw angle moves past `bin_size/2 + hysteresis`
(default 4°) from the *previous* bin's center, then snaps to the new nearest
bin. One-sided noise near a boundary no longer produces spurious bin changes.

## Event-driven capture (run-length collapsing at the source)

A camera holding steady at 30fps produces dozens of near-identical frames
while a pose is held. `MovementExtractor` emits a `Movement` only when the
quantized vector actually changes, and yields `None` otherwise — so
everything downstream of it sees the sequence of *distinct key poses*,
typically 5-20 per gesture, not hundreds.

That one decision serves both directions at once. A recording is the
sequence of movements the recorder collected, so it is run-length-collapsed
by construction rather than by a separate filtering pass. And live matching
only wakes up on a bin change instead of on every camera frame — a big
constant-factor win before any algorithmic optimization, and the reason the
matcher's rolling window is measured in pose changes rather than in
seconds.

## Matching: DTW over quantized event sequences

A gesture is a *sequence* of pose changes in a specific order, not a single
point — the same "blob" of angle-values could be visited in the wrong order
by an unrelated motion. `gestures/matcher.py`'s `dtw_distance(a, b, bin_size)` runs classic dynamic
time warping between two quantized sequences:

```
cost(i, j) = step_cost(a[i], b[j]) + min(cost(i-1,j), cost(i,j-1), cost(i-1,j-1))
step_cost(x, y) = mean(|x - y|) / bin_size        # "average bins off" for that step
```

normalized by `(n + m)` (total path length) so the score is comparable
across templates of different lengths, regardless of how fast/slow the
gesture was performed (DTW's elastic alignment absorbs speed differences)
while still penalizing pose changes done out of order.

### Cost of a match, and two things that cut it

Every step cost is non-negative, so the cheapest cell of a DP row is a lower
bound on the final one: once that bound is worse than the best score seen so
far, the template has already lost and the remaining rows cannot rescue it.
`dtw_distance` takes an `abandon_above` bound and returns `inf` at that
point. This is a pruning, not an approximation — it can only ever drop
templates that would have scored above the bound, so the winner is
unchanged. In practice most templates abandon in the first row or two.

The DP itself runs on Python floats rather than on a numpy array. The
recurrence is serial in `j`, so it cannot be vectorized along a row, and at
these sizes (a template is a handful of poses) element-wise indexing into a
2-D numpy array costs several times more per cell than the arithmetic does.
What *is* vectorized is the step-cost matrix, computed for all `(i, j)` in
one broadcast rather than one row at a time. `gestures/angles.py` is
vectorized for the same reason: 15 joint angles as a handful of whole-array
operations, not one small numpy call per joint.

`GestureMatcher` keeps a rolling buffer of live quantized events (capped at
2x the longest loaded template) and, on every new event, DTW-scores the
buffer's tail against each template not currently in cooldown. The
best-scoring template under `threshold` (default `0.6` average bins off)
fires, and a per-template cooldown (default 1000ms) prevents one motion from
triggering twice.

`threshold` is intentionally not a fixed constant: false positives (an
unrelated motion scoring low enough to match) and false negatives (a real
gesture performed slightly differently missing the threshold) trade off
against each other, and where that tradeoff should sit depends on the
specific set of recorded gestures — there's no one correct value. `apps/desktop.py`
exposes it live via `[`/`]` (tighten/loosen, shown on-screen) instead of
requiring a restart, since dialing it in is inherently an iterate-while-
testing process: trigger the false positive, tighten until it stops,
confirm the real gesture still matches.

### Known limitation

This compares the buffer as a whole against each template rather than doing
proper subsequence/online DTW — it's an approximation, not an exact
subsequence match. Nothing here is speed-sensitive: recordings store no
timing at all (see
`docs/adr/0001-recording-format-stores-no-timing.md`), so a gesture
performed at any speed scores the same. Adding speed-sensitive matching
would mean reintroducing a timestamp to the stored format, not just reading
one that is already there.

## Cursor mode: centroid tracking, dead zone, EMA

`cursor/screen.py` turns the hand's position into a screen position
(`cursor/driver.py` is what actually moves the cursor).
Three problems, each addressed independently:

### Why the centroid, not a fingertip

Tracking a single fingertip (e.g. the index tip) ties cursor position
directly to a point that moves a lot as fingers articulate through a
gesture — pointing, then curling into a fist, visibly drags the cursor.
`HandCenterExtractor` instead averages all 21 image-space landmarks. The
centroid moves far less than any individual fingertip as the hand changes
pose, which is what makes it possible to run cursor mode and gesture
matching *at the same time* off the same hand, rather than needing to
switch between "position" and "shape" tracking.

### Dead zone: hysteresis on raw position, not just smoothed output

An EMA filter (below) reduces jitter but never fully eliminates it: under
continued per-frame landmark noise the smoothed value keeps chasing a
jittering target instead of settling. `ScreenPointConverter` first checks the
incoming raw `(x, y)` against the last *committed* position and ignores the
update entirely if the movement is under `deadzone` (default `0.008`,
normalized frame units) — the same Schmitt-trigger idea as
`quantize_angles()`'s hysteresis, applied to position instead of angle
bins. Only a move that clears the dead zone becomes the new committed point
and proceeds to remapping and smoothing; a held hand yields `None` and the
cursor is never told to move at all. This is what actually makes
the cursor hold rock-still when the hand is stationary, rather than merely
reducing the jitter's amplitude.

### Region remap + EMA smoothing

Of moves that do clear the dead zone: raw camera-frame coordinates are
first remapped from a central region of the frame (`region_margin` from
each edge, default `0.2`) to the full `[0,1]` range, clamped — so a
comfortable hand range covers the whole screen instead of requiring the
hand to reach the physical frame edges. The remapped position is then
EMA-smoothed (`smoothing`, default `0.35` — lower is smoother but laggier)
before being scaled to screen pixels and handed on as a `ScreenPoint`.
`CursorDriver` is the only thing that turns one of those into a
`pyautogui.moveTo()` — the arithmetic above never touches the OS, which is
what lets the web front-end run the identical pipeline and simply draw the
point instead of moving a real cursor.
