from collections.abc import Callable

import pyautogui

from .config import ActionsConfig

pyautogui.FAILSAFE = True


class ActionDispatcher:
    """The fixed set of OS effects a matched gesture can fire. A gesture
    whose name maps to nothing is still matched and reported - it is simply
    not a macro."""

    def __init__(self, config: ActionsConfig) -> None:
        scroll = config.scroll_amount
        self._actions: dict[str, Callable[[], None]] = {
            "left-click": lambda: pyautogui.click(button="left"),
            "right-click": lambda: pyautogui.click(button="right"),
            "double-click": lambda: pyautogui.doubleClick(),
            "scroll-up": lambda: pyautogui.scroll(scroll),
            "scroll-down": lambda: pyautogui.scroll(-scroll),
        }

    @property
    def names(self) -> list[str]:
        return list(self._actions)

    def dispatch(self, name: str) -> bool:
        """Fires the action mapped to a gesture name. Returns False if
        there is no action for that name (detection-only)."""
        action = self._actions.get(name)
        if action is None:
            return False
        action()
        return True
