#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CACHE_DIR="${CACHE_DIR:-.cache}"

if [[ -x "${VENV_DIR}/bin/python" && -x "${VENV_DIR}/bin/pip" ]]; then
  "${VENV_DIR}/bin/python" -c "import sys; print(sys.executable)" >/dev/null
  "${VENV_DIR}/bin/pip" --version >/dev/null
else
  if [[ -d "${VENV_DIR}" ]]; then
    rm -rf "${VENV_DIR}"
  fi
  mkdir -p "${CACHE_DIR}"
  "${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"

  GETPIP_URL="https://bootstrap.pypa.io/get-pip.py"
  GETPIP_PATH="${CACHE_DIR}/get-pip.py"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${GETPIP_URL}" -o "${GETPIP_PATH}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${GETPIP_PATH}" "${GETPIP_URL}"
  else
    echo "ERROR: need curl or wget to download get-pip.py" >&2
    exit 1
  fi

  "${VENV_DIR}/bin/python" "${GETPIP_PATH}" --disable-pip-version-check
fi

"${VENV_DIR}/bin/pip" install --upgrade "pip==24.2" "setuptools==75.6.0" "wheel==0.43.0"
"${VENV_DIR}/bin/pip" install -r requirements.txt
"${VENV_DIR}/bin/pip" install -e .

echo "Venv ready: ${VENV_DIR}"
