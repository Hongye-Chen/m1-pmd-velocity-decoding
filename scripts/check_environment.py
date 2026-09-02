"""Check that the activated environment provides everything the project needs.

Run from the repository root with the project environment active::

    python -m scripts.check_environment

Every direct dependency and the project package are imported, and installed
versions are reported. A failed import exits with status 1.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from importlib import import_module, metadata


@dataclass(frozen=True)
class Requirement:
    """A direct dependency of the project."""

    distribution: str
    module: str
    role: str


REQUIRED_PACKAGES: tuple[Requirement, ...] = (
    Requirement("numpy", "numpy", "runtime"),
    Requirement("scipy", "scipy", "runtime"),
    Requirement("pandas", "pandas", "runtime"),
    Requirement("scikit-learn", "sklearn", "runtime"),
    Requirement("pynwb", "pynwb", "runtime"),
    Requirement("h5py", "h5py", "runtime"),
    Requirement("torch", "torch", "runtime"),
    Requirement("matplotlib", "matplotlib", "runtime"),
    Requirement("pyyaml", "yaml", "runtime"),
    Requirement("pytest", "pytest", "testing"),
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of importing one dependency."""

    requirement: Requirement
    installed_version: str | None
    error: str | None

    @property
    def importable(self) -> bool:
        return self.error is None

    @property
    def status(self) -> str:
        if not self.importable:
            return "MISSING"
        return "ok"


def check_packages(
    requirements: tuple[Requirement, ...] = REQUIRED_PACKAGES,
) -> list[CheckResult]:
    """Import every direct requirement and report its installed version."""
    results: list[CheckResult] = []
    for requirement in requirements:
        error: str | None = None
        version: str | None = None
        try:
            module = import_module(requirement.module)
        except Exception as exc:  # noqa: BLE001 - reported, not silenced
            error = f"{type(exc).__name__}: {exc}"
        else:
            version = _installed_version(requirement.distribution, module)
        results.append(
            CheckResult(
                requirement=requirement,
                installed_version=version,
                error=error,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-torch-device",
        action="store_true",
        help="Do not query PyTorch for CUDA device information.",
    )
    args = parser.parse_args(argv)

    print(f"Python           {sys.version.split()[0]}")
    print(f"Interpreter      {sys.executable}")

    results = check_packages()
    width = max(len(result.requirement.distribution) for result in results)
    print("\nDependencies")
    for result in results:
        version = result.installed_version or "-"
        print(
            f"  {result.requirement.distribution:<{width}}  {version:<12}"
            f"  {result.requirement.role:<8}  {result.status}"
        )
        if result.error is not None:
            print(f"      import failed: {result.error}")

    print("\nProject package")
    try:
        import neural_decoding
    except Exception as exc:  # noqa: BLE001 - reported, not silenced
        print(f"  neural_decoding import failed: {type(exc).__name__}: {exc}")
        print("  Run this check from the repository root.")
        return 1
    print(f"  neural_decoding {neural_decoding.__version__}")

    config = neural_decoding.load_config()
    print(f"  config sources: {', '.join(str(path) for path in config.sources)}")
    nwb_file = config.paths.nwb_file
    if nwb_file is None:
        print("  NWB file: not configured (see README.md)")
    else:
        state = "found" if nwb_file.is_file() else "MISSING"
        print(f"  NWB file: {nwb_file} ({state})")

    if not args.skip_torch_device:
        _report_torch_device()

    missing = [result for result in results if not result.importable]
    if missing:
        names = ", ".join(result.requirement.distribution for result in missing)
        print(f"\nFAILED: could not import {names}")
        return 1

    print("\nPASSED: every direct dependency imports")
    return 0


def _installed_version(distribution: str, module: object) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return getattr(module, "__version__", None)


def _report_torch_device() -> None:
    print("\nPyTorch device")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - reported, not silenced
        print(f"  unavailable: {type(exc).__name__}: {exc}")
        return
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA build:     {torch.version.cuda or 'cpu-only'}")
    if torch.cuda.is_available():
        print(f"  device 0:       {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    raise SystemExit(main())
