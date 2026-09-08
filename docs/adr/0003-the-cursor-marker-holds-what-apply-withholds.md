# The cursor marker holds what `apply` withholds

`CursorPipeline.apply` returns `None` whenever the dead zone holds the hand,
which is most frames of a hand that is merely still. The Streamlit demo draws
a Cursor Marker into the video frame, and a marker drawn only on the frames
that report a point strobes off and on at the rate the hand jitters. We gave
`CursorPipeline` a `marker` property that holds the last reported point until
the hand leaves the frame or cursor mode is switched off, and left `apply`
returning `None` exactly as before.

The pipeline therefore reports, through one member, a position the other
withholds. That is surprising enough to write down, because the obvious
"fix" — making `apply` repeat the held point — is a two-line change that
breaks the desktop app: `CursorDriver` would issue a `pyautogui.moveTo` every
frame for a motionless hand, which is the thing the dead zone exists to
prevent, and the suite would still pass.

## Considered options

**Holding in `recv()`** keeps the pipeline's two members consistent, at the
cost of putting the rule in `apps/`, which is omitted from the coverage gate
— so the rule that keeps the marker steady would have been the one piece of
it never tested. It also means the desktop app cannot adopt the marker later
without copying the four lines.

**A separate `HeldCursorPipeline`** gives the property its own type, but two
types differing by one field is an inheritance tax for a boolean's worth of
behaviour, and the front-end still has to choose between them at construction.

## Consequences

`marker` is the front-end-facing answer to "where is the cursor now"; the
return of `apply` is the answer to "where should the cursor be moved to", and
those are genuinely different questions. A front-end that draws asks the
first; one that drives the OS asks the second.

The desktop app can adopt the marker with no new logic — it is left alone
today because the real cursor is already its feedback, and a second pointer
in the camera window lagging the real one by the dead zone reads as two
competing cursors.

Anyone tempted to make the two agree should change the drawing side, not the
returning side.
