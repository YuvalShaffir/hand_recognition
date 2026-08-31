import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config.json")


@dataclass
class PathsConfig:
    model_path: str = "hand_landmarker.task"
    recordings_dir: str = "recordings"


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480


@dataclass
class LandmarkerConfig:
    num_hands: int = 1
    min_hand_detection_confidence: float = 0.6
    min_hand_presence_confidence: float = 0.7
    min_tracking_confidence: float = 0.6


@dataclass
class QuantizeConfig:
    bin_size_deg: float = 15.0
    hysteresis_deg: float = 4.0


@dataclass
class MatchConfig:
    threshold_default: float = 0.6
    threshold_step: float = 0.05
    threshold_min: float = 0.1
    threshold_max: float = 2.0
    cooldown_ms: int = 1000


@dataclass
class CursorConfig:
    region_margin: float = 0.2
    smoothing: float = 0.35
    deadzone: float = 0.008


@dataclass
class ActionsConfig:
    scroll_amount: int = 120


@dataclass
class KeybindConfig:
    quit: str = "q"
    toggle_cursor: str = "c"
    toggle_record: str = "r"
    threshold_tighten: str = "["
    threshold_loosen: str = "]"


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


def _section(section_cls: type, overrides: dict[str, Any]):
    valid = {f.name for f in fields(section_cls)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"unknown {section_cls.__name__} key(s): {sorted(unknown)}")
    return section_cls(**overrides)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Loads AppConfig from a JSON file, falling back to the default value
    for any section/key the file omits or doesn't have. A missing file
    yields all defaults (same as AppConfig())."""
    path = Path(path)
    if not path.exists():
        return AppConfig()

    data = json.loads(path.read_text())
    unknown_sections = set(data) - {f.name for f in fields(AppConfig)}
    if unknown_sections:
        raise ValueError(f"unknown config section(s): {sorted(unknown_sections)}")

    kwargs = {f.name: _section(f.type, data.get(f.name, {})) for f in fields(AppConfig)}
    return AppConfig(**kwargs)
