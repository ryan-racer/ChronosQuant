"""Basic package sanity tests (no model download required)."""

import chronosquant


def test_version() -> None:
    assert chronosquant.__version__


def test_chronos2_pipeline_importable() -> None:
    from chronos import Chronos2Pipeline  # noqa: F401
