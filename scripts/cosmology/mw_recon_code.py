from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_FFTW_VERSION = "3.3.10"
_FFTW_URL = f"https://www.fftw.org/fftw-{_FFTW_VERSION}.tar.gz"


# 関数: `_sha256_file` の入出力契約と処理意図を定義する。
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# 関数: `_run` の入出力契約と処理意図を定義する。

def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=None if (cwd is None) else str(cwd),
        env=None if (env is None) else {**os.environ, **env},
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# 関数: `_ensure_git_clone` の入出力契約と処理意図を定義する。

def _ensure_git_clone(*, root: Path, url: str, dst: Path) -> None:
    # 条件分岐: `dst.exists()` を満たす経路を評価する。
    if dst.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", url, str(dst)], cwd=root)


# 関数: `_ensure_fftw` の入出力契約と処理意図を定義する。

def _ensure_fftw(*, root: Path, deps_dir: Path, jobs: int, force: bool) -> tuple[Path, Path]:
    tar_path = deps_dir / f"fftw-{_FFTW_VERSION}.tar.gz"
    src_dir = deps_dir / f"fftw-{_FFTW_VERSION}"
    build_dir = deps_dir / f"fftw-{_FFTW_VERSION}-build"
    install_dir = deps_dir / f"fftw-{_FFTW_VERSION}-install"
    inc_dir = install_dir / "include"
    lib_dir = install_dir / "lib"

    # 条件分岐: `(not force) and (inc_dir / "fftw3.h").exists() and (lib_dir / "libfftw3.so")....` を満たす経路を評価する。
    if (not force) and (inc_dir / "fftw3.h").exists() and (lib_dir / "libfftw3.so").exists():
        return inc_dir, lib_dir

    deps_dir.mkdir(parents=True, exist_ok=True)

    # 条件分岐: `(not tar_path.exists()) or force` を満たす経路を評価する。
    if (not tar_path.exists()) or force:
        _run(["bash", "-lc", f"curl -L -o '{tar_path}' '{_FFTW_URL}'"], cwd=root)

    # 条件分岐: `src_dir.exists() and force` を満たす経路を評価する。

    if src_dir.exists() and force:
        _run(["bash", "-lc", f"rm -rf '{src_dir}'"], cwd=root)

    # 条件分岐: `build_dir.exists() and force` を満たす経路を評価する。

    if build_dir.exists() and force:
        _run(["bash", "-lc", f"rm -rf '{build_dir}'"], cwd=root)

    # 条件分岐: `install_dir.exists() and force` を満たす経路を評価する。

    if install_dir.exists() and force:
        _run(["bash", "-lc", f"rm -rf '{install_dir}'"], cwd=root)

    # 条件分岐: `not src_dir.exists()` を満たす経路を評価する。

    if not src_dir.exists():
        _run(["bash", "-lc", f"tar -xf '{tar_path}' -C '{deps_dir}'"], cwd=root)

    build_dir.mkdir(parents=True, exist_ok=True)

    cfg = (
        f"'{src_dir}/configure' "
        f"--prefix='{install_dir}' "
        "--enable-openmp --enable-threads --disable-fortran "
        "--enable-shared --enable-static"
    )
    _run(["bash", "-lc", cfg], cwd=build_dir)
    _run(["bash", "-lc", f"make -j{int(jobs)}"], cwd=build_dir)
    _run(["bash", "-lc", "make install"], cwd=build_dir)

    # 条件分岐: `not (inc_dir / "fftw3.h").exists()` を満たす経路を評価する。
    if not (inc_dir / "fftw3.h").exists():
        raise RuntimeError("FFTW build failed (missing fftw3.h)")

    # 条件分岐: `not (lib_dir / "libfftw3.so").exists()` を満たす経路を評価する。

    if not (lib_dir / "libfftw3.so").exists():
        raise RuntimeError("FFTW build failed (missing libfftw3.so)")

    return inc_dir, lib_dir


# 関数: `_ensure_recon_mw_binary` の入出力契約と処理意図を定義する。

def _ensure_recon_mw_binary(
    *,
    root: Path,
    recon_code_dir: Path,
    fftw_inc: Path,
    fftw_lib: Path,
    jobs: int,
    force: bool,
) -> Path:
    build_dir = recon_code_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    bin_path = build_dir / "recon_mw"
    stamp_path = build_dir / "recon_mw_build_stamp.json"

    driver = root / "scripts" / "cosmology" / "mw_recon_driver.cpp"
    # 条件分岐: `not driver.exists()` を満たす経路を評価する。
    if not driver.exists():
        raise RuntimeError("mw_recon_driver.cpp is missing (internal error)")

    driver_sha256 = _sha256_file(driver)
    # 条件分岐: `bin_path.exists() and (not force) and stamp_path.exists()` を満たす経路を評価する。
    if bin_path.exists() and (not force) and stamp_path.exists():
        try:
            stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
            # 条件分岐: `isinstance(stamp, dict) and stamp.get("driver_sha256") == driver_sha256` を満たす経路を評価する。
            if isinstance(stamp, dict) and stamp.get("driver_sha256") == driver_sha256:
                return bin_path
        except Exception:
            pass

    srcs = [
        str(driver),
        str(recon_code_dir / "io.cpp"),
        str(recon_code_dir / "multigrid.cpp"),
        str(recon_code_dir / "grid.cpp"),
        str(recon_code_dir / "shift.cpp"),
        str(recon_code_dir / "smooth.cpp"),
    ]
    cmd = [
        "bash",
        "-lc",
        " ".join(
            [
                "g++",
                "-O3",
                "-fopenmp",
                "-DSKIPRAW",
                "-DREADWEIGHT",
                f"-I'{recon_code_dir}'",
                f"-I'{fftw_inc}'",
                *[f"'{s}'" for s in srcs],
                f"-L'{fftw_lib}'",
                f"-Wl,-rpath,'{fftw_lib}'",
                "-lfftw3_omp",
                "-lfftw3",
                "-lm",
                f"-o '{bin_path}'",
            ]
        ),
    ]
    _run(cmd, cwd=build_dir, env={"OMP_NUM_THREADS": str(int(jobs))})

    # 条件分岐: `not bin_path.exists()` を満たす経路を評価する。
    if not bin_path.exists():
        raise RuntimeError("recon_mw build failed (binary missing)")

    try:
        stamp = {
            "driver_sha256": driver_sha256,
            "fftw_version": _FFTW_VERSION,
            "compile": {"jobs": int(jobs), "cmd": " ".join(cmd)},
        }
        stamp_path.write_text(json.dumps(stamp, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
    except Exception:
        pass

    return bin_path


# 関数: `_write_ascii_catalog` の入出力契約と処理意図を定義する。

def _write_ascii_catalog(path: Path, ra: np.ndarray, dec: np.ndarray, col3: np.ndarray, w: np.ndarray, *, col3_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# ra dec {str(col3_name)} weight\n")
        n = int(ra.size)
        chunk = 200_000
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            block = np.column_stack(
                [
                    np.asarray(ra[i:j], dtype=np.float64),
                    np.asarray(dec[i:j], dtype=np.float64),
                    np.asarray(col3[i:j], dtype=np.float64),
                    np.asarray(w[i:j], dtype=np.float64),
                ]
            )
            np.savetxt(f, block, fmt="%.8f %.8f %.8f %.8f")


# 関数: `_xyz_to_radec_dist` の入出力契約と処理意図を定義する。

def _xyz_to_radec_dist(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = xyz[:, 0].astype(np.float64, copy=False)
    y = xyz[:, 1].astype(np.float64, copy=False)
    z = xyz[:, 2].astype(np.float64, copy=False)
    dist = np.sqrt(x * x + y * y + z * z)
    ra = np.degrees(np.arctan2(y, x))
    ra = np.mod(ra + 360.0, 360.0)
    dec = np.degrees(np.arcsin(np.clip(z / (dist + 1e-30), -1.0, 1.0)))
    return ra, dec, dist


# 関数: `run_mw_recon` の入出力契約と処理意図を定義する。

def run_mw_recon(
    *,
    root: Path,
    ra_g: np.ndarray,
    dec_g: np.ndarray,
    z_g: np.ndarray,
    dist_g: np.ndarray | None = None,
    w_g_recon: np.ndarray,
    ra_r: np.ndarray,
    dec_r: np.ndarray,
    z_r: np.ndarray,
    dist_r: np.ndarray | None = None,
    w_r_recon: np.ndarray,
    bias: float,
    f_growth: float,
    smoothing_mpc_over_h: float,
    omega_m: float,
    nthreads: int,
    random_rsd: bool,
    force_rebuild: bool,
    input_mode: str = "z",
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    # 条件分岐: `os.name == "nt"` を満たす経路を評価する。
    if os.name == "nt":
        raise RuntimeError("mw_multigrid recon is supported only on Linux/WSL")

    input_mode = str(input_mode)
    # 条件分岐: `input_mode not in ("z", "chi")` を満たす経路を評価する。
    if input_mode not in ("z", "chi"):
        raise ValueError("input_mode must be z/chi")

    # 条件分岐: `input_mode == "z"` を満たす経路を評価する。

    if input_mode == "z":
        col3_g = np.asarray(z_g, dtype=np.float64)
        col3_r = np.asarray(z_r, dtype=np.float64)
        col3_name = "z"
    else:
        # 条件分岐: `dist_g is None or dist_r is None` を満たす経路を評価する。
        if dist_g is None or dist_r is None:
            raise ValueError("chi mode requires dist_g/dist_r")

        col3_g = np.asarray(dist_g, dtype=np.float64)
        col3_r = np.asarray(dist_r, dtype=np.float64)
        col3_name = "chi_mpc_over_h"

    recon_code_dir = root / ".tmp_recon_code"
    _ensure_git_clone(root=root, url="https://github.com/martinjameswhite/recon_code.git", dst=recon_code_dir)

    deps_dir = recon_code_dir / "_deps"
    fftw_inc, fftw_lib = _ensure_fftw(root=root, deps_dir=deps_dir, jobs=int(nthreads), force=bool(force_rebuild))
    recon_bin = _ensure_recon_mw_binary(
        root=root,
        recon_code_dir=recon_code_dir,
        fftw_inc=fftw_inc,
        fftw_lib=fftw_lib,
        jobs=int(nthreads),
        force=bool(force_rebuild),
    )

    with tempfile.TemporaryDirectory(prefix="mw_recon_") as td:
        tdir = Path(td)
        f_data = tdir / "data.txt"
        f_rnd = tdir / "random.txt"
        _write_ascii_catalog(f_data, ra_g, dec_g, col3_g, w_g_recon, col3_name=col3_name)
        _write_ascii_catalog(f_rnd, ra_r, dec_r, col3_r, w_r_recon, col3_name=col3_name)

        env = {"OMP_NUM_THREADS": str(int(nthreads))}
        cmd = [
            str(recon_bin),
            str(f_data),
            str(f_rnd),
            f"{float(bias):.8g}",
            f"{float(f_growth):.8g}",
            f"{float(smoothing_mpc_over_h):.8g}",
            f"{float(omega_m):.8g}",
            "1" if bool(random_rsd) else "0",
            str(input_mode),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(tdir),
            env={**os.environ, **env},
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        data_out = np.loadtxt(tdir / "data_rec.xyzw", dtype=np.float64)
        rnd_out = np.loadtxt(tdir / "rand_rec.xyzw", dtype=np.float64)
        # 条件分岐: `data_out.ndim != 2 or data_out.shape[1] < 3` を満たす経路を評価する。
        if data_out.ndim != 2 or data_out.shape[1] < 3:
            raise RuntimeError("mw recon output invalid: data_rec.xyzw")

        # 条件分岐: `rnd_out.ndim != 2 or rnd_out.shape[1] < 3` を満たす経路を評価する。

        if rnd_out.ndim != 2 or rnd_out.shape[1] < 3:
            raise RuntimeError("mw recon output invalid: rand_rec.xyzw")

        ra_g2, dec_g2, d_g2 = _xyz_to_radec_dist(np.asarray(data_out[:, :3], dtype=np.float64))
        ra_r2, dec_r2, d_r2 = _xyz_to_radec_dist(np.asarray(rnd_out[:, :3], dtype=np.float64))

    meta: dict[str, Any] = {
        "backend": "mw_multigrid",
        "recon_code": {
            "repo_dir": str(recon_code_dir),
            "binary": str(recon_bin),
            "fftw": {"version": _FFTW_VERSION, "inc": str(fftw_inc), "lib": str(fftw_lib)},
        },
        "spec": {
            "bias": float(bias),
            "f_growth": float(f_growth),
            "smoothing_mpc_over_h": float(smoothing_mpc_over_h),
            "omega_m": float(omega_m),
            "random_rsd": bool(random_rsd),
            "input_mode": str(input_mode),
        },
        "runtime": {
            "stdout_tail": "\n".join(str(proc.stdout).splitlines()[-30:]),
            "stderr_tail": "\n".join(str(proc.stderr).splitlines()[-30:]),
        },
    }
    return (ra_g2, dec_g2, d_g2), (ra_r2, dec_r2, d_r2), meta
