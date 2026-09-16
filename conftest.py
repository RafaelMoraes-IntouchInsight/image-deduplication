"""Makes the package importable when tests are run from the repo root.

Without this, `pytest tests/` cannot import image_deduplication: pytest puts the
test file's own directory on sys.path, not the project root, so the suite only
worked under `python -m pytest` (which adds the CWD) or after `pip install -e .`.
"""
