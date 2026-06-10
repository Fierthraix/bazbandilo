#!/usr/bin/env python3

import json
from pathlib import Path
import sys


CWD: Path = Path(__file__).parent


if __name__ == "__main__":
    file1 = CWD / "results_fam_inflated_almost_all.json"
    file2 = CWD / "results_fam_inflated_last_two.json"

    output_file = CWD / "results_fam_inflated_full.json"

    if output_file.is_file() and output_file.stat().st_size != 00:
        print(f"Outfile {output_file} exists already.")
        sys.exit(1)

    with file1.open("r") as f:
        results1 = json.load(f)
    assert isinstance(results1, list)

    with file2.open("r") as f:
        results2 = json.load(f)
    assert isinstance(results2, list)

    results1.extend(results2)

    with output_file.open("w") as f:
        json.dump(results1, f)
