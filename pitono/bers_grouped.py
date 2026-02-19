#!/usr/bin/env python
from plot import GROUP_MARKERS, GROUPS, base_parser, plot_bers, load_json, set_snr_xlim

from argparse import Namespace
import re
from typing import Dict, List


def parse_args() -> Namespace:
    ap = base_parser()
    ap.add_argument("-r", "--regex", default="", type=str)
    ap.add_argument("-g", "--group", type=int, choices=(1, 2, 3), default=None)
    ap.add_argument("--ebn0", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    import gc
    import matplotlib.pyplot as plt

    args = parse_args()
    set_snr_xlim(args.snr_db_min, args.snr_db_max)
    if args.group:
        group_ids = [args.group]
    else:
        group_ids = [1, 2, 3]

    regex = re.compile(args.regex)
    bers: List[Dict[str, object]] = load_json(args.ber_file, filter=regex)
    grouped_bers: List[List[Dict]] = [
        [b for b in bers if b["name"] in GROUPS[group_id]] for group_id in group_ids
    ]
    del bers
    gc.collect()

    for group_id, bers in zip(group_ids, grouped_bers):
        if args.ebn0:
            name: str = f"bers_ebn0_group_{group_id}"
        else:
            name: str = f"bers_snr_group_{group_id}"
        plot_bers(
            bers,
            ebn0=args.ebn0,
            save_path=args.save_dir / name,
            cycles=GROUP_MARKERS[group_id],
        )

    if not args.save:
        plt.show()
