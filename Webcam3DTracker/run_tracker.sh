#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_python() {
    local candidate version

    if [[ -n "${PYTHON_BIN:-}" ]]; then
        if [[ -x "${PYTHON_BIN}" ]]; then
            echo "${PYTHON_BIN}"
            return 0
        fi
        if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
            echo "${PYTHON_BIN}"
            return 0
        fi
        echo "[launcher] PYTHON_BIN is set but not executable: ${PYTHON_BIN}" >&2
        return 1
    fi

    for candidate in python3.12 python3.11 python3.10 python3; do
        if ! command -v "${candidate}" >/dev/null 2>&1; then
            continue
        fi

        version="$(${candidate} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
        if [[ "${version}" == "3.10" || "${version}" == "3.11" || "${version}" == "3.12" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    echo "[launcher] No supported Python 3.10-3.12 interpreter was found." >&2
    return 1
}

PYTHON_CMD="$(find_python)"
exec "${PYTHON_CMD}" "${SCRIPT_DIR}/launch_tracker.py" "$@"