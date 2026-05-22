#!/usr/bin/env python3
"""Check that CUDA extension modules used by FlashAvatar are importable."""

from __future__ import annotations

import sys


def main() -> int:
    import torch  # noqa: F401  # preload libtorch before CUDA extensions

    ok = True
    for label, mod in (
        ("simple_knn (KNN distances)", "simple_knn._C"),
        ("diff_gaussian_rasterization", "diff_gaussian_rasterization._C"),
    ):
        try:
            __import__(mod)
            print(f"OK  {label}: import {mod}")
        except Exception as e:
            ok = False
            print(f"FAIL {label}: {e}", file=sys.stderr)

    if not ok:
        print(
            "\nInstall from repo root (torch must already be installed):\n"
            "  python -m pip install --no-build-isolation -e submodules/simple-knn\n"
            "  python -m pip install --no-build-isolation -e submodules/diff-gaussian-rasterization\n",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
