#!/usr/bin/env python
from plot import base_parser, plot_bers, load_json

from argparse import Namespace
import matplotlib.pyplot as plt
import re


def parse_args() -> Namespace:
    ap = base_parser()
    ap.add_argument("-r", "--regex", default="", type=str)
    ap.add_argument("--ebn0", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    regex = re.compile(args.regex)

    bers = load_json(args.ber_file, filter=regex)

    plot_bers(
        bers,
        ebn0=args.ebn0,
        save_path=args.save_dir / ("bers_ebn0.png" if args.ebn0 else "bers_snr.png"),
    )

    if not args.save:
        plt.show()
