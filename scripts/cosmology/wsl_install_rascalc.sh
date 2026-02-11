#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GSL_VER="${GSL_VER:-2.7.1}"
DEPS_DIR="${ROOT_DIR}/.tmp_rascalc_deps"
PREFIX="${DEPS_DIR}/prefix"
BUILD_ROOT="${DEPS_DIR}/build"
LOG_DIR="${DEPS_DIR}/logs"
SRC_DIR="${ROOT_DIR}/data/cosmology/sources/gnu"
TARBALL="${SRC_DIR}/gsl-${GSL_VER}.tar.gz"

mkdir -p "${SRC_DIR}" "${BUILD_ROOT}" "${LOG_DIR}" "${PREFIX}"

echo "[RascalC] ROOT_DIR=${ROOT_DIR}"
echo "[RascalC] PREFIX=${PREFIX}"

if [[ ! -f "${TARBALL}" ]]; then
  echo "[RascalC] Downloading GSL ${GSL_VER} -> ${TARBALL}"
  curl -L --retry 3 --retry-delay 2 -o "${TARBALL}" "https://ftp.gnu.org/gnu/gsl/gsl-${GSL_VER}.tar.gz"
else
  echo "[RascalC] Using cached ${TARBALL}"
fi

echo "[RascalC] Building GSL (static) ..."
cd "${BUILD_ROOT}"
rm -rf "gsl-${GSL_VER}"
tar -xzf "${TARBALL}" -C "${BUILD_ROOT}"

cd "gsl-${GSL_VER}"
./configure --prefix="${PREFIX}" --disable-shared --enable-static > "${LOG_DIR}/gsl_configure.log" 2>&1 || {
  tail -n 120 "${LOG_DIR}/gsl_configure.log"
  exit 1
}

JOBS="$(command -v nproc >/dev/null 2>&1 && nproc || getconf _NPROCESSORS_ONLN || echo 1)"
make -j"${JOBS}" > "${LOG_DIR}/gsl_build.log" 2>&1 || {
  tail -n 120 "${LOG_DIR}/gsl_build.log"
  exit 1
}
make install > "${LOG_DIR}/gsl_install.log" 2>&1 || {
  tail -n 120 "${LOG_DIR}/gsl_install.log"
  exit 1
}

echo "[RascalC] Installed GSL version: $("${PREFIX}/bin/gsl-config" --version)"

echo "[RascalC] Installing minimal pkg-config wrapper (for gsl) ..."
mkdir -p "${PREFIX}/bin"
cat > "${PREFIX}/bin/pkg-config" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ge 2 && "$1" == "--cflags" && "$2" == "gsl" ]]; then
  exec "$(dirname "$0")/gsl-config" --cflags
fi
if [[ $# -ge 2 && "$1" == "--libs" && "$2" == "gsl" ]]; then
  exec "$(dirname "$0")/gsl-config" --libs
fi
if [[ $# -ge 2 && "$1" == "--modversion" && "$2" == "gsl" ]]; then
  exec "$(dirname "$0")/gsl-config" --version
fi
echo "pkg-config wrapper: unsupported args: $*" >&2
exit 2
EOF
chmod +x "${PREFIX}/bin/pkg-config"

echo "[RascalC] pkg-config --cflags gsl: $("${PREFIX}/bin/pkg-config" --cflags gsl)"
echo "[RascalC] pkg-config --libs   gsl: $("${PREFIX}/bin/pkg-config" --libs gsl)"

echo "[RascalC] Installing Python deps into .venv_wsl ..."
cd "${ROOT_DIR}"
.venv_wsl/bin/python -m pip install -U pip
.venv_wsl/bin/python -m pip install -U astropy

echo "[RascalC] Installing RascalC (from local cached source) ..."
export PATH="${PREFIX}/bin:${PATH}"
.venv_wsl/bin/python -m pip install -U "${ROOT_DIR}/data/cosmology/sources/github/RascalC"

echo "[RascalC] Done. Verify with: .venv_wsl/bin/python -c \"import RascalC; print(RascalC.__version__)\""
