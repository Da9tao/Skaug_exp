"""Cross-check effective diffusivity maps produced by the C++ utility."""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, Tuple

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

import numpy as np
from scipy import ndimage

T_TO_SIZE = {
    1000: 247.1080942,
    2000: 216.5761537,
    3000: 204.049624,
    4000: 191.427262,
    5000: 184.5860421,
    6000: 174.8961444,
    7000: 169.1660005,
    8000: 166.2407307,
    9000: 161.038961,
    10000: 156.051519,
    30000: 120.5969061,
    50000: 105.975122,
    70000: 98.08307894,
    100000: 87.42639647,
    200000: 72.80066049,
    400000: 59.41720823,
    700000: 50.42732327,
}


def load_grid(path: pathlib.Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D grid in {path}, got shape {data.shape}")
    return data


def extract_time_stamp(path: pathlib.Path) -> int:
    name = path.stem
    idx = name.find("t=")
    if idx == -1:
        raise ValueError(f"Unable to parse time stamp from {path}")
    idx += 2
    end = idx
    while end < len(name) and name[end].isdigit():
        end += 1
    return int(name[idx:end])


def compute_gradients(dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    grad_x = ndimage.sobel(dist, axis=1, mode="nearest") / 8.0
    grad_y = ndimage.sobel(dist, axis=0, mode="nearest") / 8.0
    return grad_x, grad_y


def compute_diffusivities(
    dist: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    r_ref: float,
    D0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-12
    flat_dist = dist.ravel()
    flat_gx = grad_x.ravel()
    flat_gy = grad_y.ravel()

    Dp = np.zeros_like(flat_dist)
    Dv = np.zeros_like(flat_dist)

    valid = flat_dist > eps
    if not np.any(valid):
        return (
            Dp.reshape(dist.shape),
            Dv.reshape(dist.shape),
            Dp.reshape(dist.shape),
            Dp.reshape(dist.shape),
        )

    R_over_d = np.zeros_like(flat_dist)
    R_over_d[valid] = r_ref / flat_dist[valid]
    d_over_R = np.zeros_like(flat_dist)
    d_over_R[valid] = flat_dist[valid] / r_ref

    inner = valid & (R_over_d <= 2.0)
    outer = valid & (~inner)

    if np.any(inner):
        Rz = R_over_d[inner]
        Rz2 = Rz * Rz
        Rz3 = Rz2 * Rz
        Rz4 = Rz3 * Rz
        Rz5 = Rz4 * Rz
        Dp[inner] = D0 * (
            1.0
            - 0.5625 * Rz
            + 0.125 * Rz3
            - 45.0 * Rz4 / 256.0
            - Rz5 / 6.0
        )

    if np.any(outer):
        logz = np.log(d_over_R[outer])
        denom = logz * logz - 4.325 * logz + 1.591
        valid_denom = np.abs(denom) > eps
        tmp = np.zeros_like(logz)
        tmp[valid_denom] = -D0 * (2.0 * (logz[valid_denom] - 0.9543)) / denom[valid_denom]
        Dp[outer] = tmp

    numerator = 6.0 * d_over_R[valid] * d_over_R[valid] + 2.0 * d_over_R[valid]
    denominator = 6.0 * d_over_R[valid] * d_over_R[valid] + 9.0 * d_over_R[valid] + 2.0
    valid_denom = np.abs(denominator) > eps
    Dv_vals = np.zeros_like(numerator)
    Dv_vals[valid_denom] = D0 * (numerator[valid_denom] / denominator[valid_denom])
    Dv[valid] = Dv_vals

    grad_sq = flat_gx * flat_gx + flat_gy * flat_gy
    grad_valid = grad_sq > eps

    De_x = np.zeros_like(flat_dist)
    De_y = np.zeros_like(flat_dist)

    if np.any(grad_valid):
        inv_mag = 1.0 / np.sqrt(grad_sq[grad_valid])
        nx = flat_gx[grad_valid] * inv_mag
        ny = flat_gy[grad_valid] * inv_mag

        cos_x = 1.0 - nx * nx
        cos_y = 1.0 - ny * ny

        De_x[grad_valid] = Dp[grad_valid] * cos_x + Dv[grad_valid] * (1.0 - cos_x)
        De_y[grad_valid] = Dp[grad_valid] * cos_y + Dv[grad_valid] * (1.0 - cos_y)

    zero_grad = valid & (~grad_valid)
    if np.any(zero_grad):
        De_x[zero_grad] = Dp[zero_grad]
        De_y[zero_grad] = Dp[zero_grad]

    return (
        Dp.reshape(dist.shape),
        Dv.reshape(dist.shape),
        De_x.reshape(dist.shape),
        De_y.reshape(dist.shape),
    )


def rms_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def plot_comparison(
    dist: np.ndarray,
    cpp_maps: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    py_maps: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output: pathlib.Path,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    labels = ["Dp", "Dv", "De_x", "De_y"]
    fig, axes = plt.subplots(len(labels), 3, figsize=(15, 4 * len(labels)))

    for row, label in enumerate(labels):
        py_map = py_maps[row]
        cpp_map = cpp_maps[row]
        diff = cpp_map - py_map

        im0 = axes[row, 0].imshow(py_map, cmap="viridis")
        axes[row, 0].set_title(f"Python {label}")
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(cpp_map, cmap="viridis")
        axes[row, 1].set_title(f"C++ {label}")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im2 = axes[row, 2].imshow(diff, cmap="seismic")
        axes[row, 2].set_title(f"Difference {label}")
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def iter_distance_maps(path: pathlib.Path) -> Iterable[pathlib.Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix == ".txt")
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Path {path} does not exist")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dist_path",
        type=pathlib.Path,
        help="Distance-map file or directory",
    )
    parser.add_argument(
        "cpp_output_dir",
        type=pathlib.Path,
        help="Directory containing the C++ diffusivity outputs",
    )
    parser.add_argument(
        "--particle-size",
        type=float,
        default=200.0,
        help="Particle diameter in nm",
    )
    parser.add_argument(
        "--d0",
        type=float,
        default=0.02,
        help="Reference diffusivity in um^2/s",
    )
    parser.add_argument(
        "--figure-dir",
        type=pathlib.Path,
        default=pathlib.Path("de_comparisons"),
        help="Directory for comparison figures",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip figure generation (useful when matplotlib is unavailable)",
    )

    args = parser.parse_args()

    dist_paths = list(iter_distance_maps(args.dist_path))
    if not dist_paths:
        raise FileNotFoundError(f"No distance maps found at {args.dist_path}")

    if not args.cpp_output_dir.exists():
        raise FileNotFoundError(
            f"C++ output directory {args.cpp_output_dir} does not exist"
        )

    if not args.no_figure:
        if plt is None:
            raise RuntimeError(
                "matplotlib is not installed; re-run with --no-figure to skip plots"
            )
        args.figure_dir.mkdir(parents=True, exist_ok=True)

    for dist_path in dist_paths:
        dist = load_grid(dist_path)
        if dist.shape[0] != dist.shape[1]:
            raise ValueError(f"Distance map {dist_path} is not square: {dist.shape}")
        rows = dist.shape[0]
        time_stamp = extract_time_stamp(dist_path)
        if time_stamp not in T_TO_SIZE:
            raise KeyError(f"Time stamp {time_stamp} missing from t_to_size map")
        domain_size_um = T_TO_SIZE[time_stamp]
        grid_size_um = domain_size_um / float(rows)
        particle_radius_um = 0.5 * args.particle_size * 1e-3
        r_ref = particle_radius_um / grid_size_um

        grad_x, grad_y = compute_gradients(dist)
        py_maps = compute_diffusivities(dist, grad_x, grad_y, r_ref, args.d0)

        base_name = dist_path.stem
        cpp_maps = (
            load_grid(args.cpp_output_dir / f"{base_name}_Dp.txt"),
            load_grid(args.cpp_output_dir / f"{base_name}_Dv.txt"),
            load_grid(args.cpp_output_dir / f"{base_name}_De_x.txt"),
            load_grid(args.cpp_output_dir / f"{base_name}_De_y.txt"),
        )

        for label, cpp_map, py_map in zip(
            ["Dp", "Dv", "De_x", "De_y"], cpp_maps, py_maps
        ):
            if cpp_map.shape != dist.shape or py_map.shape != dist.shape:
                raise ValueError(
                    f"Shape mismatch for {label} in {base_name}: dist {dist.shape}, "
                    f"cpp {cpp_map.shape}, python {py_map.shape}"
                )

        if not args.no_figure:
            figure_path = args.figure_dir / f"{base_name}_comparison.png"
            plot_comparison(dist, cpp_maps, py_maps, figure_path)

        rms_values = {
            label: rms_error(cpp_map, py_map)
            for label, cpp_map, py_map in zip(["Dp", "Dv", "De_x", "De_y"], cpp_maps, py_maps)
        }
        print(
            f"Processed {dist_path.name}: "
            + ", ".join(f"{label} RMS={value:.3e}" for label, value in rms_values.items())
        )


if __name__ == "__main__":
    main()
