"""`urllib.request.urlretrieve` faked; no network."""

import urllib.error

import pytest

from hand_recognition.vision import model_asset
from hand_recognition.vision.model_asset import MODEL_URL, ensure_model


def test_existing_file_is_not_downloaded(tmp_path, mocker):
    path = tmp_path / "hand_landmarker.task"
    path.write_bytes(b"already here")
    retrieve = mocker.patch.object(model_asset.urllib.request, "urlretrieve")

    ensure_model(str(path))

    retrieve.assert_not_called()
    assert path.read_bytes() == b"already here"


def test_missing_file_triggers_a_download_to_the_given_path(tmp_path, mocker, capsys):
    path = tmp_path / "hand_landmarker.task"

    def fake_retrieve(url, destination):
        assert url == MODEL_URL
        destination.write_bytes(b"model bytes")

    mocker.patch.object(
        model_asset.urllib.request, "urlretrieve", side_effect=fake_retrieve
    )

    ensure_model(str(path))

    assert path.read_bytes() == b"model bytes"


def test_a_url_error_raises_runtimeerror_with_manual_instructions(tmp_path, mocker):
    """The one failure a fresh clone hits offline, so the message has to
    name both the URL and the path."""
    path = tmp_path / "hand_landmarker.task"
    mocker.patch.object(
        model_asset.urllib.request,
        "urlretrieve",
        side_effect=urllib.error.URLError("offline"),
    )

    with pytest.raises(RuntimeError) as error:
        ensure_model(str(path))

    assert MODEL_URL in str(error.value)
    assert str(path) in str(error.value)


def test_a_partial_download_leaves_no_valid_looking_file(tmp_path, mocker):
    """`urlretrieve` writes as it goes, so an interrupted download must not
    leave a truncated file that `path.exists()` accepts forever after."""
    path = tmp_path / "hand_landmarker.task"

    def fail_midway(url, destination):
        destination.write_bytes(b"half a model")
        raise urllib.error.URLError("connection reset")

    mocker.patch.object(
        model_asset.urllib.request, "urlretrieve", side_effect=fail_midway
    )

    with pytest.raises(RuntimeError):
        ensure_model(str(path))

    assert list(tmp_path.iterdir()) == []


def test_the_model_url_is_https():
    """An executable artefact fetched over an unauthenticated channel is
    an executable artefact an attacker picks."""
    assert MODEL_URL.startswith("https://")
