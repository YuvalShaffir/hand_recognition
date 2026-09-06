import pyautogui

from ..domain import ScreenPoint

pyautogui.FAILSAFE = True


class CursorDriver:
    """Moves the real OS cursor. The only part of cursor control that
    touches `pyautogui`, so everything upstream stays importable where
    there is no desktop to drive."""

    def __init__(self) -> None:
        size = pyautogui.size()
        self._size: tuple[int, int] = (size[0], size[1])

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._size

    def move_to(self, point: ScreenPoint) -> None:
        pyautogui.moveTo(point.x, point.y, _pause=False)
