import pytest


def _data_available() -> bool:
    try:
        from fml.paths import data_dir
        data_dir()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _data_available():
        return
    skip = pytest.mark.skip(reason="private data repo not found (set FML_DATA_DIR)")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)
