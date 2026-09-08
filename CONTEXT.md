# Hand Recognition

Webcam hand tracking that lets a user record a hand gesture, name it, and have
a mapped OS mouse action fire whenever that gesture is performed again live.
The hand can also drive the OS cursor directly, at the same time.

## Language

### Hand

**Landmark**:
One of the 21 points MediaPipe reports for a detected hand. *Image-space*
landmarks are positions within the camera frame; *world* landmarks are
positions in real-world space relative to the hand itself.
_Avoid_: point, keypoint, joint (a joint is where two bones meet, not a point).

**Pose**:
The shape of a hand at one instant, expressed as the angles at its finger
joints — independent of where the hand is, how big it is, or how it is turned
relative to the camera.
_Avoid_: position, orientation, hand state.

**Quantized Pose**:
A pose snapped to a grid of fixed-width angle bins, so that small tremors and
camera noise leave it unchanged. The hand is considered to have moved only
when its quantized pose changes.
_Avoid_: binned pose, snapped pose, discretized pose.

**Movement**:
A quantized pose that differs from the one before it — the unit of change in
the system. A hand held still produces no movements at all, however many
camera frames pass, so everything downstream counts changes rather than
frames.
_Avoid_: event, transition, frame (a frame is a camera image, and the two
deliberately do not correspond one-to-one).

### Gestures

**Gesture**:
A hand's shape changing over time, as a sequence of movements. Holding still
adds nothing to the sequence, so a gesture is a record of changes, not of
elapsed frames — and it carries no timing, so the same gesture performed
quickly or slowly is the same gesture.
_Avoid_: motion, sign. Not _movement_ either: a movement is one step of a
gesture, not the whole of it.

**Recording**:
The act of capturing a gesture from the live camera in order to name it. A
recording is something the user *does*; what it produces is a gesture
template.
_Avoid_: using this for the stored result — that is a gesture template.

**Gesture Template**:
A named, stored gesture, kept as the reference that live gestures are compared
against.
_Avoid_: sample, example, saved gesture, recording.

**Gesture Library**:
Every gesture template the system knows, together with the ability to match a
live gesture against them and to admit new ones. The authority on what a
gesture is called: it resolves a requested name into the name a template
actually ends up with.
_Avoid_: template store, gesture set, registry, collection.

### Matching

**Match**:
The judgement that a live gesture is the same gesture as some template,
allowing for the two having been performed at different speeds.
_Avoid_: recognition, detection, hit.

**Match Threshold**:
How close a live gesture must come to a template to count as a match, measured
in angle bins. Lower is stricter. It is a single value shared by every
template, adjustable while the app is running.
_Avoid_: sensitivity, tolerance, confidence (there is no probability here).

### Output

**Action**:
An effect on the operating system — a left click, a right click, a scroll.
Actions exist independently of gestures; the set of them is fixed.
_Avoid_: command, event, effect.

**Macro**:
A gesture template bound to an action, so that matching the gesture fires the
action. A template whose name matches no known action is still matched and
reported — it is simply not a macro.
_Avoid_: binding, mapping, shortcut, hotkey.

**Cursor Mode**:
Driving a cursor from the hand's position in the camera frame — the real OS
cursor on the desktop, a drawn Cursor Marker in the browser. It runs
independently of matching, and the two can be active at once.
_Avoid_: mouse mode, pointer mode, tracking mode.

**Cursor Marker**:
A drawn stand-in for the cursor, showing where cursor mode is pointing
without an OS cursor to move. It persists while a hand is in frame,
including while the dead zone is holding the hand still.
_Avoid_: fake cursor, pointer, crosshair.

### Machinery

These are not domain terms — they are the vocabulary the code is built out
of, listed here because every module is named in them.

**Stage**:
One step of the work, taking a single item and returning a single item. A
stage that has nothing to report returns nothing rather than skipping, so a
stage never falls out of step with the camera frames driving it.
_Avoid_: filter, transform, node, handler.

**Pipeline**:
Stages composed into one longer stage, which is itself a stage. The user-
facing ones are the gesture pipeline (hands in, gesture names out) and the
cursor pipeline (hands in, screen positions out).
_Avoid_: chain, graph, flow.

**Fork**:
Feeding one item to two stages and pairing what they return, so a single
stream of hands drives gesture matching and cursor control at once without
either being aware of the other.
_Avoid_: split, tee, branch.
