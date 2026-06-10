#!/usr/bin/env python3

import json
from pathlib import Path
import re


CWD: Path = Path(__file__).parent


if __name__ == "__main__":

    merged_file = CWD / "results_merged.json"

    json_files = sorted(
        (f for f in CWD.glob("*.json") if f != merged_file), key=lambda j: j.name
    )

    merged_contents = []

    for file in json_files:
        with file.open("r") as f:
            if re.search(r'^results_9[0-9]{2}_.+\.json', file.name):
                merged_contents.extend(reversed(json.load(f)))
            else:
                merged_contents.extend(json.load(f))
            print(f"Loaded {file.name}")

    with merged_file.open("w") as f:
        print(f"Saving to {merged_file.name}")
        json.dump(merged_contents, f)
