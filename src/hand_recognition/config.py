import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, cast

DEFAULT_CONFIG_PATH = Path("config.json")

# Recordings store angles as bin *indices*, so a vanishingly small bin makes
# an ordinary angle a huge index - and an angle finer than this is noise off
# `arccos`, not hand movement.
MIN_BIN_SIZE_DEG = 0.1

# MediaPipe tracks a handful of hands at most; anything beyond this is a
# configuration mistake or an attempt to make one frame cost minutes.
MAX_NUM_HANDS = 4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass
class PathsConfig:
    model_path: str = "hand_landmarker.task"
    recordings_dir: str = "recordings"

    def __post_init__(self) -> None:
        _require(bool(self.model_path), "paths.model_path must not be empty")
        _require(bool(self.recordings_dir), "paths.recordings_dir must not be empty")


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        _require(self.index >= 0, "camera.index must not be negative")
        _require(self.width > 0, "camera.width must be positive")
        _require(self.height > 0, "camera.height must be positive")


@dataclass
class LandmarkerConfig:
    num_hands: int = 1
    min_hand_detection_confidence: float = 0.6
    min_hand_presence_confidence: float = 0.7
    min_tracking_confidence: float = 0.6

    def __post_init__(self) -> None:
        _require(
            1 <= self.num_hands <= MAX_NUM_HANDS,
            f"landmarker.num_hands must be between 1 and {MAX_NUM_HANDS}",
        )
        for name in (
            "min_hand_detection_confidence",
            "min_hand_presence_confidence",
            "min_tracking_confidence",
        ):
            _require(
                0.0 <= getattr(self, name) <= 1.0,
                f"landmarker.{name} must be between 0 and 1",
            )


@dataclass
class QuantizeConfig:
    bin_size_deg: float = 15.0
    hysteresis_deg: float = 4.0

    def __post_init__(self) -> None:
        _require(
            self.bin_size_deg >= MIN_BIN_SIZE_DEG,
            f"quantize.bin_size_deg must be at least {MIN_BIN_SIZE_DEG}",
        )
        _require(
            self.hysteresis_deg >= 0, "quantize.hysteresis_deg must not be negative"
        )


@dataclass
class MatchConfig:
    threshold_default: float = 0.6
    threshold_step: float = 0.05
    threshold_min: float = 0.1
    threshold_max: float = 2.0
    cooldown_ms: int = 1000

    def __post_init__(self) -> None:
        _require(self.threshold_min > 0, "match.threshold_min must be positive")
        _require(
            self.threshold_min <= self.threshold_max,
            "match.threshold_min must not exceed match.threshold_max",
        )
        _require(
            self.threshold_min <= self.threshold_default <= self.threshold_max,
            "match.threshold_default must lie between the threshold bounds",
        )
        _require(self.threshold_step > 0, "match.threshold_step must be positive")
        _require(self.cooldown_ms >= 0, "match.cooldown_ms must not be negative")


@dataclass
class CursorConfig:
    region_margin: float = 0.2
    smoothing: float = 0.35
    deadzone: float = 0.008

    def __post_init__(self) -> None:
        _require(
            0.0 <= self.region_margin < 0.5,
            "cursor.region_margin must be at least 0 and below 0.5",
        )
        _require(
            0.0 <= self.smoothing <= 1.0, "cursor.smoothing must be between 0 and 1"
        )
        _require(self.deadzone >= 0, "cursor.deadzone must not be negative")


@dataclass
class ActionsConfig:
    scroll_amount: int = 120

    def __post_init__(self) -> None:
        _require(self.scroll_amount >= 0, "actions.scroll_amount must not be negative")


@dataclass
class KeybindConfig:
    quit: str = "q"
    toggle_cursor: str = "c"
    toggle_record: str = "r"
    threshold_tighten: str = "["
    threshold_loosen: str = "]"

    def __post_init__(self) -> None:
        for f in fields(self):
            _require(
                len(getattr(self, f.name)) == 1,
                f"keybindings.{f.name} must be a single character",
            )


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    landmarker: LandmarkerConfig = field(default_factory=LandmarkerConfig)
    quantize: QuantizeConfig = field(default_factory=QuantizeConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    keybindings: KeybindConfig = field(default_factory=KeybindConfig)


def _value(section_cls: type, key: str, value: Any, expected: type) -> Any:
    """JSON is untyped, so a value's type is checked here rather than left to
    surface much later inside numpy or cv2. `type(...) is` rather than
    `isinstance`, because `bool` is an `int` and `True` is not a width."""
    if expected is float and type(value) is int:
        return float(value)
    if type(value) is not expected:
        raise ValueError(
            f"{section_cls.__name__}.{key} must be {expected.__name__}, "
            f"got {value!r}"
        )
    return value


def _section(section_cls: type, overrides: dict[str, Any]):
    valid = {f.name: cast(type, f.type) for f in fields(section_cls)}
    unknown = set(overrides) - set(valid)
    if unknown:
        raise ValueError(f"unknown {section_cls.__name__} key(s): {sorted(unknown)}")
    typed = {
        key: _value(section_cls, key, value, valid[key])
        for key, value in overrides.items()
    }
    return section_cls(**typed)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Loads AppConfig from a JSON file, falling back to the default value
    for any section/key the file omits or doesn't have. A missing file
    yields all defaults (same as AppConfig())."""
    path = Path(path)
    if not path.exists():
        return AppConfig()

    data = json.loads(path.read_text(encoding="utf-8"))
    unknown_sections = set(data) - {f.name for f in fields(AppConfig)}
    if unknown_sections:
        raise ValueError(f"unknown config section(s): {sorted(unknown_sections)}")

    kwargs = {
        f.name: _section(cast(type, f.type), data.get(f.name, {}))
        for f in fields(AppConfig)
    }
    return AppConfig(**kwargs)
