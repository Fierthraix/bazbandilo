#!/usr/bin/env python3
from util import timeit, db, undb
from plot import (
    DETECTORS,
    base_parser,
    load_json,
    multi_parse,
    save_figure,
)

from argparse import Namespace
from collections import defaultdict
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional


def calculate_pd(pfa: float, h0_λs: List[float], h1_λs: List[float]) -> float:
    assert 0 <= pfa <= 1
    λ0: float = get_threshold(pfa, h0_λs)
    return np.mean([λ > λ0 for λ in h1_λs])


def get_threshold(pfa: float, h0_λs: List[float]):
    return np.quantile(h0_λs, 1 - pfa)


def parse_results(
    modulation: Dict[str, object],
    pfas: List[float] = [0.1, 0.05, 0.01],
) -> Dict[str, object]:
    mod_res = {
        "name": modulation["name"],
        "snrs": modulation["snrs"],
    }
    for dx_result in modulation["results"]:
        dx = {
            "kind": dx_result["kind"],
            "h0_λs": dx_result["h0_λs"],
            "h1_λs": dx_result["h1_λs"],
            "pfas": defaultdict(list),
            "λ0s": defaultdict(list),
        }
        for h0_λ, h1_λ in zip(dx["h0_λs"], dx["h1_λs"]):  # For each SNR.
            for pfa in pfas:
                dx["pfas"][pfa].append(calculate_pd(pfa, h0_λ, h1_λ))
                dx["λ0s"][pfa].append(get_threshold(pfa, h0_λ))
        mod_res[dx_result["kind"]] = dx
    return mod_res


def get_thresholds(
    pfa: float, regressed: List[Dict[str, object]], detector: str
) -> List[Tuple[str, List[float]]]:
    return [
        (modulation["name"], modulation[detector]["λ0s"][pfa])
        for modulation in regressed
    ]


def parse_args() -> Namespace:
    ap = base_parser()
    ap.add_argument("-f", "--pfa", default=[0.01], type=float, nargs="+")
    ap.add_argument("-r", "--regex", default="", type=str)
    return ap.parse_args()


def generate_ber_pd_mesh(
    snrs_linear,
    bers,
    pds,
    snr_diff_min_db=-30,
    snr_diff_max_db=5,
    num_snr_points=100,
    num_diff_points=100,
):
    # Create interpolation in the linear domain
    ber_interp = interp1d(snrs_linear, bers, kind="linear", fill_value="extrapolate")
    pd_interp = interp1d(snrs_linear, pds, kind="linear", fill_value="extrapolate")

    # Create a dB-grid for SNR_BER
    snr_min_db = db(np.min(snrs_linear))
    snr_max_db = db(np.max(snrs_linear))
    snr_ber_vals_db = np.linspace(snr_min_db, snr_max_db, num_snr_points)
    # Create a dB-grid for Delta-SNR
    snr_diff_vals_db = np.linspace(snr_diff_min_db, snr_diff_max_db, num_diff_points)

    # Build the 2D mesh
    #   snr_ber_mesh_db[i, j] is SNR_BER in dB (row dimension)
    #   snr_diff_mesh_db[i, j] is Delta-SNR in dB (column dimension)
    snr_ber_mesh_db, snr_diff_mesh_db = np.meshgrid(
        snr_ber_vals_db, snr_diff_vals_db, indexing="ij"
    )

    # Convert the SNR_BER mesh to linear
    snr_ber_mesh_lin = undb(snr_ber_mesh_db)
    # SNR for detection = SNR_BER + DeltaSNR (in dB) => add, then convert to linear
    snr_pd_mesh_lin = undb(snr_ber_mesh_db + snr_diff_mesh_db)

    # Interpolate BER & PD in linear domain
    ber_mesh = ber_interp(snr_ber_mesh_lin)
    pd_mesh = pd_interp(snr_pd_mesh_lin)

    DeltaSNR_mesh = snr_diff_mesh_db
    BER_mesh = ber_mesh
    PD_mesh = pd_mesh

    return DeltaSNR_mesh, BER_mesh, PD_mesh


def plot_2d_colormap(
    DeltaSNR_mesh, BER_mesh, PD_mesh, save_path: Optional[Path] = None
):
    fig, ax = plt.subplots(figsize=(7, 5))
    pcm = ax.pcolormesh(
        DeltaSNR_mesh, BER_mesh, PD_mesh, shading="auto", cmap="viridis"
    )

    fig.colorbar(pcm, ax=ax, label="P_D")

    ax.set_xlabel("Delta SNR (dB)")
    ax.set_ylabel("BER (log scale)")
    ax.set_yscale('log')

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
        ax.set_title(save_path.stem)


def plot_2d_contour(DeltaSNR_mesh, BER_mesh, PD_mesh, save_path: Optional[Path] = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contour(DeltaSNR_mesh, BER_mesh, PD_mesh, levels=10, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=8)

    ax.set_xlabel("Delta SNR (dB)")
    ax.set_ylabel("BER (log scale)")
    # ax.set_yscale('log')

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
        ax.set_title(save_path.stem)


def plot_3d_surface(DeltaSNR_mesh, BER_mesh, PD_mesh, save_path: Optional[Path] = None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(DeltaSNR_mesh, BER_mesh, PD_mesh, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Delta SNR (dB)")
    ax.set_ylabel("BER")
    # ax.set_yscale('log')
    ax.set_zlabel("P_D")

    ax.view_init(elev=30, azim=180-45/2)

    if save_path:
        save_figure(fig, save_path)
        ax.set_title(save_path.stem)


if __name__ == "__main__":
    import gc
    import re
    import sys

    CWD: Path = Path(__file__).parent

    args = parse_args()

    print("Starting CFAR Analysis...")

    with timeit("Loading Data") as _:
        regex: re.Pattern = re.compile(args.regex)
        results: List[Dict[str, object]] = load_json(args.pd_file, filter=regex)
        gc.collect()

        bers: List[Dict[str, object]] = load_json(args.ber_file, filter=regex)
        gc.collect()

    with timeit("CFAR Analysis") as _:
        parse_fn = partial(parse_results, pfas=args.pfa)
        regressed: List[Dict[str, object]] = multi_parse(results, parse_fn)
        del results
        gc.collect()

    __dx = [k for k in regressed[0].keys() if k not in ("name", "snrs")]
    if sorted(__dx) != sorted(DETECTORS):
        DETECTORS = __dx

    with timeit("Plotting") as _:
        for pfa in args.pfa:
            for detector in DETECTORS:
                for modulation in regressed:
                    ber: List[float] = next(
                        b["bers"] for b in bers if b["name"] == modulation["name"]
                    )
                    pds: List[float] = modulation[detector]["pfas"][pfa][: len(ber)]
                    snrs: List[float] = modulation["snrs"][: len(ber)]

                    assert len(pds) == len(snrs)
                    assert len(ber) == len(snrs), f"{len(bers)},  {len(snrs)}"

                    DeltaSNR_mesh, BER_mesh, PD_mesh = generate_ber_pd_mesh(
                        snrs,
                        ber,
                        pds,
                        snr_diff_min_db=-30,
                        snr_diff_max_db=5,
                        num_snr_points=200,
                        num_diff_points=200,
                    )

                    # Make the plots
                    plot_2d_colormap(
                        DeltaSNR_mesh,
                        BER_mesh,
                        PD_mesh,
                        save_path=args.save_dir
                        / f"2D ColorMap: PD vs ΔSNR vs BER - {modulation["name"]} - {detector} - PFA={pfa}.png",
                    )
                    # plot_2d_contour(
                    #     DeltaSNR_mesh,
                    #     BER_mesh,
                    #     PD_mesh,
                    #     save_path=args.save_dir
                    #     / f"2D Contour: PD vs ΔSNR vs BER - {modulation["name"]} - {detector} - PFA={pfa}.png",
                    # )
                    plot_3d_surface(
                        DeltaSNR_mesh,
                        BER_mesh,
                        PD_mesh,
                        save_path=args.save_dir
                        / f"3D Surface: PD vs ΔSNR vs BER - {modulation["name"]} - {detector} - PFA={pfa}.png",
                    )

    if not args.save:
        plt.show()

    if not sys.flags.interactive:
        plt.close("all")
