from pathlib import Path

import pytest

import fetch_artifacts
import weather_service
from config import ARTIFACT_DIR, BEST_MODEL_PATH, CONTEXT_PATH


def test_packaged_artifacts_are_used_without_downloading(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Los artefactos locales son válidos; no se debe descargar nada.")

    monkeypatch.setattr(fetch_artifacts, "urlopen", no_network)
    assert BEST_MODEL_PATH == ARTIFACT_DIR / "best_weather_risk_model.pt"
    assert CONTEXT_PATH == ARTIFACT_DIR / "climate_context.npz"
    fetch_artifacts.verify()
    fetch_artifacts.fetch()
    assert weather_service.artifacts_ready()


def test_lfs_pointer_is_not_a_usable_artifact(tmp_path: Path, monkeypatch):
    model = tmp_path / BEST_MODEL_PATH.name
    model.write_text("version https://git-lfs.github.com/spec/v1\n")
    monkeypatch.setattr(weather_service, "BEST_MODEL_PATH", model)
    monkeypatch.setattr(fetch_artifacts, "DEST", tmp_path)
    assert not weather_service.artifacts_ready()
    with pytest.raises(RuntimeError, match=model.name):
        fetch_artifacts.verify()
