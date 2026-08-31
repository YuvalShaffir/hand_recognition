import pyautogui

pyautogui.FAILSAFE = True


def build_actions(scroll_amount: int = 120) -> dict:
    return {
        "left-click": lambda: pyautogui.click(button="left"),
        "right-click": lambda: pyautogui.click(button="right"),
        "double-click": lambda: pyautogui.doubleClick(),
        "scroll-up": lambda: pyautogui.scroll(scroll_amount),
        "scroll-down": lambda: pyautogui.scroll(-scroll_amount),
    }


ACTIONS = build_actions()


def run_action(name: str, actions: dict = ACTIONS) -> bool:
    """Run the OS action mapped to a recording name. Returns False if
    there's no action registered for that name (detection-only)."""
    action = actions.get(name)
    if action is None:
        return False
    action()
    return True
