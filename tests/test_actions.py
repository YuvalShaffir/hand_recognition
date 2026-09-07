import pytest

import pyautogui
from hand_recognition.actions import ActionDispatcher
from hand_recognition.config import ActionsConfig


@pytest.fixture
def dispatcher():
    return ActionDispatcher(ActionsConfig())


def test_dispatch_left_click_calls_pyautogui_click_left(dispatcher, mocker):
    click = mocker.patch.object(pyautogui, "click")

    dispatcher.dispatch("left-click")

    click.assert_called_once_with(button="left")


def test_dispatch_right_click_calls_pyautogui_click_right(dispatcher, mocker):
    click = mocker.patch.object(pyautogui, "click")

    dispatcher.dispatch("right-click")

    click.assert_called_once_with(button="right")


def test_dispatch_double_click_calls_pyautogui_double_click(dispatcher, mocker):
    double = mocker.patch.object(pyautogui, "doubleClick")

    dispatcher.dispatch("double-click")

    double.assert_called_once_with()


def test_scroll_up_uses_the_configured_amount(mocker):
    scroll = mocker.patch.object(pyautogui, "scroll")

    ActionDispatcher(ActionsConfig(scroll_amount=42)).dispatch("scroll-up")

    scroll.assert_called_once_with(42)


def test_scroll_down_negates_the_amount(mocker):
    scroll = mocker.patch.object(pyautogui, "scroll")

    ActionDispatcher(ActionsConfig(scroll_amount=42)).dispatch("scroll-down")

    scroll.assert_called_once_with(-42)


def test_dispatch_returns_true_when_an_action_fired(dispatcher, mocker):
    mocker.patch.object(pyautogui, "click")

    assert dispatcher.dispatch("left-click") is True


def test_names_lists_the_five_known_actions(dispatcher):
    assert dispatcher.names == [
        "left-click",
        "right-click",
        "double-click",
        "scroll-up",
        "scroll-down",
    ]


def test_unknown_name_returns_false_and_fires_nothing(dispatcher, mocker):
    click = mocker.patch.object(pyautogui, "click")
    scroll = mocker.patch.object(pyautogui, "scroll")

    assert dispatcher.dispatch("wave") is False
    click.assert_not_called()
    scroll.assert_not_called()


def test_empty_name_returns_false(dispatcher):
    assert dispatcher.dispatch("") is False


def test_dispatch_propagates_a_pyautogui_failsafe(dispatcher, mocker):
    """The failsafe is a deliberate user abort; swallowing it would take
    away the only way to stop a program that is moving the cursor."""
    mocker.patch.object(
        pyautogui, "click", side_effect=pyautogui.FailSafeException("aborted")
    )

    with pytest.raises(pyautogui.FailSafeException):
        dispatcher.dispatch("left-click")


def test_names_are_case_sensitive(dispatcher, mocker):
    click = mocker.patch.object(pyautogui, "click")

    assert dispatcher.dispatch("Left-Click") is False
    click.assert_not_called()


def test_zero_scroll_amount_still_calls_scroll(mocker):
    scroll = mocker.patch.object(pyautogui, "scroll")

    ActionDispatcher(ActionsConfig(scroll_amount=0)).dispatch("scroll-up")

    scroll.assert_called_once_with(0)


def test_failsafe_is_enabled_on_import():
    assert pyautogui.FAILSAFE is True
