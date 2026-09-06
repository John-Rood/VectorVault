from importlib.metadata import version
from pathlib import Path
import re
import warnings

from google.genai import types


ROOT = Path(__file__).resolve().parents[1]


def _numeric_version(value):
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    assert match, f"Expected a three-part numeric version, got {value!r}"
    return tuple(int(part) for part in match.groups())


def test_canonical_release_metadata_and_dependency_floors():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "7\.4\.9\.23"$', pyproject, re.MULTILINE)
    assert re.search(r'^requires-python = ">=3\.10"$', pyproject, re.MULTILINE)
    assert '"google-genai>=1.56.0"' in pyproject
    assert not (ROOT / "setup.py").exists()

    lock_path = ROOT / "uv.lock"
    if lock_path.is_file():
        lockfile = lock_path.read_text(encoding="utf-8")
        assert re.search(r'^requires-python = ">=3\.10"$', lockfile, re.MULTILINE)
        assert re.search(
            r'^name = "vector-vault"\nversion = "7\.4\.9\.23"$',
            lockfile,
            re.MULTILINE,
        )
        assert '{ name = "google-genai", specifier = ">=1.56.0" }' in lockfile


def test_declared_google_genai_floor_supports_every_gemini_level():
    assert _numeric_version(version("google-genai")) >= (1, 56, 0)
    assert "thinking_level" in types.ThinkingConfig.model_fields

    for level in ("MINIMAL", "LOW", "MEDIUM", "HIGH"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=level)
            )
            serialized = config.model_dump(exclude_none=True, mode="json")
        assert caught == []
        assert serialized == {"thinking_config": {"thinking_level": level}}
