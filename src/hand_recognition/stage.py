from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Generic, TypeVar

In = TypeVar("In")
Out = TypeVar("Out")
Next = TypeVar("Next")
Side = TypeVar("Side")


class Stage(ABC, Generic[In, Out]):
    """One step of a pipeline. A stage turns a single item into a single
    item (`apply`) and, by extension, a stream into a stream (calling it).
    Stages compose left to right with `|`."""

    @abstractmethod
    def apply(self, item: In) -> Out: ...

    def __call__(self, source: Iterator[In]) -> Iterator[Out]:
        return (self.apply(item) for item in source)

    def __or__(self, other: "Stage[Out, Next]") -> "Stage[In, Next]":
        return _Chain(self, other)


class _Chain(Stage[In, Out], Generic[In, Next, Out]):
    def __init__(self, first: Stage[In, Next], second: Stage[Next, Out]) -> None:
        self._first = first
        self._second = second

    def apply(self, item: In) -> Out:
        return self._second.apply(self._first.apply(item))


class OptionalStage(Stage[In | None, Out | None]):
    """A stage that only acts on real items: `None` passes straight through.

    Every stage stays one-item-in/one-item-out this way, even the ones that
    only produce a result some of the time (no hand in frame, hand held
    still, gesture not matched), so a pipeline never falls out of step with
    the camera frames driving it."""

    def apply(self, item: In | None) -> Out | None:
        return None if item is None else self.transform(item)

    @abstractmethod
    def transform(self, item: In) -> Out | None: ...


class Fork(Stage[In, tuple[Out, Side]], Generic[In, Out, Side]):
    """Feeds each item to two stages and pairs their results, so one hand
    stream drives gesture recognition and cursor control at the same time.

    The branches are independent and could run concurrently, but they are
    far too lopsided to be worth a thread: the cursor branch is a hundredth
    of the gesture branch, and the pure-Python DTW loop holds the GIL
    anyway, so a worker costs more in handoff than the overlap saves.
    """

    def __init__(self, first: Stage[In, Out], second: Stage[In, Side]) -> None:
        self._first = first
        self._second = second

    def apply(self, item: In) -> tuple[Out, Side]:
        return self._first.apply(item), self._second.apply(item)
