from __future__ import annotations

import importlib.util
import subprocess
import sys


def is_installed(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except ModuleNotFoundError:
        return False


def install() -> None:
    if not is_installed("segment_anything"):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "segment-anything"],
            check=True,
        )


try:
    import launch

    skip_install = launch.args.skip_install
except Exception:
    skip_install = False

if not skip_install:
    install()
