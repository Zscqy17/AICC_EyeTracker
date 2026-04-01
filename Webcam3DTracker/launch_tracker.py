from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_DIR / "requirements.txt"
TRACKER_FILE = PROJECT_DIR / "MonitorTracking.py"
VENV_DIR = PROJECT_DIR / ".venv"
STAMP_FILE = VENV_DIR / ".requirements.sha256"


def log(message: str) -> None:
    print(f"[launcher] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the environment and launch the eye tracker.")
    parser.add_argument("camera_index", nargs="?", type=int, help="Optional camera index to pass to the tracker.")
    parser.add_argument("--setup-only", action="store_true", help="Prepare the virtual environment and install dependencies, then exit.")
    parser.add_argument("--force-recreate", action="store_true", help="Delete and recreate the virtual environment before installing dependencies.")
    return parser.parse_args()


def supported_runtime() -> bool:
    return sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12


def requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS_FILE.read_bytes()).hexdigest()


def venv_python() -> Path:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    python_name = "python.exe" if os.name == "nt" else "python"
    return VENV_DIR / scripts_dir / python_name


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    display = " ".join(command)
    log(display)
    subprocess.run(command, cwd=PROJECT_DIR, env=env, check=True)


def python_minor_version(python_path: Path) -> tuple[int, int] | None:
    if not python_path.exists():
        return None

    result = subprocess.run(
        [str(python_path), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    version_text = result.stdout.strip()
    try:
        major_text, minor_text = version_text.split(".", maxsplit=1)
        return int(major_text), int(minor_text)
    except ValueError:
        return None


def version_is_supported(version: tuple[int, int] | None) -> bool:
    return version is not None and version[0] == 3 and 10 <= version[1] <= 12


def environment_is_healthy(python_path: Path) -> bool:
    if not version_is_supported(python_minor_version(python_path)):
        return False

    check_code = (
        "import importlib; "
        "modules = ('cv2', 'numpy', 'mediapipe', 'scipy', 'pyautogui'); "
        "[importlib.import_module(name) for name in modules]; "
        "import mediapipe as mp; "
        "assert hasattr(mp, 'solutions')"
    )
    result = subprocess.run(
        [str(python_path), "-c", check_code],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def recreate_virtualenv(current_python: str) -> Path:
    if VENV_DIR.exists():
        log(f"Removing incompatible environment: {VENV_DIR}")
        shutil.rmtree(VENV_DIR)

    run([current_python, "-m", "venv", str(VENV_DIR)])
    return venv_python()


def install_requirements(python_path: Path) -> None:
    pip_env = os.environ.copy()
    pip_env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], env=pip_env)
    run([str(python_path), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], env=pip_env)
    STAMP_FILE.write_text(requirements_hash(), encoding="utf-8")


def ensure_environment(current_python: str, force_recreate: bool) -> Path:
    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing requirements file: {REQUIREMENTS_FILE}")
    if not TRACKER_FILE.exists():
        raise FileNotFoundError(f"Missing tracker entry file: {TRACKER_FILE}")

    env_python = venv_python()
    recreate = force_recreate or not environment_is_healthy(env_python)
    if recreate:
        env_python = recreate_virtualenv(current_python)

    required_hash = requirements_hash()
    installed_hash = STAMP_FILE.read_text(encoding="utf-8").strip() if STAMP_FILE.exists() else ""
    if recreate or installed_hash != required_hash or not environment_is_healthy(env_python):
        install_requirements(env_python)

    if not environment_is_healthy(env_python):
        raise RuntimeError("Environment check failed after installing dependencies.")

    return env_python


def launch_tracker(env_python: Path, camera_index: int | None) -> int:
    env = os.environ.copy()
    if camera_index is not None:
        env["CAMERA_INDEX"] = str(camera_index)
        log(f"Using CAMERA_INDEX={camera_index}")

    process = subprocess.run([str(env_python), str(TRACKER_FILE)], cwd=PROJECT_DIR, env=env)
    return process.returncode


def main() -> int:
    args = parse_args()
    if not supported_runtime():
        raise RuntimeError("This launcher must be started with Python 3.10, 3.11, or 3.12.")

    env_python = ensure_environment(sys.executable, args.force_recreate)
    log(f"Environment ready: {env_python}")

    if args.setup_only:
        log("Setup complete. Skipping tracker launch because --setup-only was provided.")
        return 0

    return launch_tracker(env_python, args.camera_index)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(1)