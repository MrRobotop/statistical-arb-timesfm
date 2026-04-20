"""Hardware and environment preflight check for PairsTrader / TimesFM 2.5."""

from __future__ import annotations

import shutil
import sys


def _status(label: str, ok: bool | None, msg: str) -> None:
    tag = "PASS" if ok is True else ("WARN" if ok is None else "FAIL")
    print(f"  [{tag:4s}] {label}: {msg}")


def check_python() -> bool:
    ver = sys.version_info
    ok = ver >= (3, 11)
    _status(
        "Python",
        ok or None if not ok else True,
        f"{ver.major}.{ver.minor}.{ver.micro}"
        + ("" if ok else " (need ≥ 3.11)"),
    )
    return ok


def check_ram() -> bool:
    try:
        import psutil
        mem = psutil.virtual_memory()
        avail_gb = mem.available / 1024 ** 3
        total_gb = mem.total / 1024 ** 3
        if avail_gb < 2.0:
            _status("RAM", False, f"{avail_gb:.1f} GB available of {total_gb:.1f} GB (need ≥ 2 GB)")
            return False
        elif avail_gb < 4.0:
            _status("RAM", None, f"{avail_gb:.1f} GB available of {total_gb:.1f} GB (warn < 4 GB)")
        else:
            _status("RAM", True, f"{avail_gb:.1f} GB available of {total_gb:.1f} GB")
        return True
    except ImportError:
        _status("RAM", None, "psutil not installed; skipping check")
        return True


def check_disk() -> bool:
    usage = shutil.disk_usage(".")
    free_gb = usage.free / 1024 ** 3
    if free_gb < 2.0:
        _status("Disk", False, f"{free_gb:.1f} GB free (need ≥ 2 GB for model weights)")
        return False
    elif free_gb < 5.0:
        _status("Disk", None, f"{free_gb:.1f} GB free (recommend ≥ 5 GB)")
    else:
        _status("Disk", True, f"{free_gb:.1f} GB free")
    return True


def check_torch() -> bool:
    try:
        import torch  # noqa: F401
        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        accel = "CUDA" if cuda else ("MPS" if mps else "CPU only")
        _status("PyTorch", True, f"{torch.__version__}  [{accel}]")
        return True
    except ImportError:
        _status("PyTorch", False, "not installed")
        return False


def check_timesfm() -> bool:
    try:
        import timesfm  # noqa: F401
        _status("timesfm", True, "importable")
        return True
    except Exception as exc:
        _status("timesfm", False, f"import failed: {exc}")
        return False


def main() -> int:
    print("─" * 50)
    print("  PairsTrader — System Preflight Check")
    print("─" * 50)

    results = [
        check_python(),
        check_ram(),
        check_disk(),
        check_torch(),
        check_timesfm(),
    ]

    print("─" * 50)
    if all(results):
        print("  Result: ALL CHECKS PASSED")
    elif any(r is False for r in results):
        fails = sum(1 for r in results if r is False)
        print(f"  Result: {fails} CHECK(S) FAILED — see FAIL lines above")
    else:
        print("  Result: PASSED WITH WARNINGS — see WARN lines above")
    print("─" * 50)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
