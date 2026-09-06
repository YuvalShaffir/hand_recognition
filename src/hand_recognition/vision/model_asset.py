import urllib.error
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker"
    "/hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model(model_path: str) -> None:
    """Downloads the hand landmarker model on first run if it isn't already
    present locally - it's an ~8MB generated binary asset, kept out of git."""
    path = Path(model_path)
    if path.exists():
        return

    print(f"{path} not found, downloading from {MODEL_URL} ...")
    # `urlretrieve` writes as it goes, so downloading straight to `path`
    # leaves an interrupted fetch looking like a valid model forever after.
    partial = path.with_name(path.name + ".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, partial)
    except urllib.error.URLError as e:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"couldn't download the hand landmarker model: {e}\n"
            f"download it manually from {MODEL_URL} and save it to {path}"
        ) from e
    partial.replace(path)
    print(f"saved {path}")
