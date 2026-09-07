from ..domain import Hand, NormalizedPoint
from ..stage import OptionalStage


class HandCenterExtractor(OptionalStage[Hand, NormalizedPoint]):
    """Hands in, hand centers out: the mean of all 21 image-space
    landmarks, rather than a single fingertip, so the cursor stays put
    while the fingers move through a gesture - letting cursor control and
    gesture recognition run at once off the same hand."""

    def transform(self, item: Hand) -> NormalizedPoint:
        n = len(item.landmarks)
        return NormalizedPoint(
            x=sum(lm.x for lm in item.landmarks) / n,
            y=sum(lm.y for lm in item.landmarks) / n,
        )
