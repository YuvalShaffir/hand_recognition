import pytest

import pyautogui
from hand_recognition.cursor.driver import CursorDriver
from hand_recognition.domain import ScreenPoint


def test_screen_size_is_read_once_at_construction(mocker):
    size = mocker.patch.object(pyautogui, "size", return_value=(800, 600))

    driver = CursorDriver()

    assert driver.screen_size == (800, 600)
    assert driver.screen_size == (800, 600)
    size.assert_called_once_with()


def test_move_to_calls_pyautogui_moveto_with_the_point(mocker):
    move = mocker.patch.object(pyautogui, "moveTo")

    CursorDriver().move_to(ScreenPoint(x=10.5, y=20.5))

    assert move.call_args.args == (10.5, 20.5)


def test_move_to_passes_pause_false(mocker):
    """`_pause=True` throttles pyautogui to ~10 Hz, which makes cursor mode
    unusable - non-obvious enough to pin."""
    move = mocker.patch.object(pyautogui, "moveTo")

    CursorDriver().move_to(ScreenPoint(x=1.0, y=2.0))

    assert move.call_args.kwargs == {"_pause": False}


def test_failsafe_is_enabled_on_import():
    assert pyautogui.FAILSAFE is True


def test_a_failsafe_exception_propagates(mocker):
    mocker.patch.object(
        pyautogui, "moveTo", side_effect=pyautogui.FailSafeException("aborted")
    )

    with pytest.raises(pyautogui.FailSafeException):
        CursorDriver().move_to(ScreenPoint(x=0.0, y=0.0))
