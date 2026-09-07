# Gesture libraries are in-memory by default

The Streamlit demo is publicly reachable and builds one `GesturePipeline` per
browser session inside a single process, so a gesture library backed by a
directory made every visitor's recordings visible to — and matchable by —
every other visitor, in both directions: a new library globbed whatever was
already on the server's disk, and admitting a template wrote into that same
shared directory. We made `GestureLibrary` in-memory and empty at
construction, and moved loading and saving out to two functions in
`gestures/persistence.py` that only the desktop app calls.

## Considered options

**Per-session temporary directories** would have preserved recordings across a
page reload, but buys that with a cleanup lifecycle that has to actually run —
and the web app already leans on `__del__` for detector teardown, which is not
a guarantee. Abnormal termination leaks directories full of user data.

**A `PersistentGestureLibrary` subclass** keeps one name for two lifetimes,
and leaves the subclass inheriting the directory-globbing constructor that
caused the leak in the first place.

## Consequences

The safe behaviour is now the default: forgetting to configure persistence
gives you isolation rather than sharing. Web recordings do not survive a page
reload, which is a real loss for a demo where recording is the first thing a
visitor does.

Name uniqueness splits in two. Uniqueness *within* a library stays the
library's job; filename collision on disk becomes `save_template`'s. The two
can disagree — a desktop library seeded from disk and a file written by
another process — and `save_template` is the one that has to cope.

`GestureLibrary` can now be tested with no filesystem at all. If one of its
tests needs `tmp_path`, the split has regressed.
