#!/usr/bin/env python3

import json
from pathlib import Path
from typing import List


CWD: Path = Path(__file__).parent


if __name__ == "__main__":

    output_file = CWD / "results_ssca_merged_100_000.json"

    fam_file: Path = CWD / "results_fam_merged_100_000.json"
    with fam_file.open("r") as f:
        print(f"Loading {fam_file.name}")
        fam_results = json.load(f)

    ssca_file: Path = CWD / "results_normal_ssca_10_000.json"
    with ssca_file.open("r") as f:
        print(f"Loading {ssca_file.name}")
        ssca_results = json.load(f)

    assert len(fam_results) == len(ssca_results)

    dx_replace: List[str] = [
        "MaxCut",
        "Dcs",
    ]
    dx_add: List[str] = [
        "Energy",
        "NormalTest",
    ]

    # fam_results: 21  <- List of Modulations
    # fam_results[0]: {'name', 'results', 'snrs'}  <- Modulation Wrapper
    # fam_results[0]['results']: 4  <- List for each detector type.
    # fam_results[0]['results'][0]: {'kind', 'snrs', 'h0_λs', 'h1_λs'}
    # mod_fam: {'name', 'results', 'snrs'}  <-
    # fam_attempt: 4  <- List for each detector type.
    # fam_attempt[0]: {'kind', 'snrs', 'h0_λs', 'h1_λs'}  <- Name field is here

    final_results: List = []
    for mod_fam, mod_ssca in zip(fam_results, ssca_results):
        assert len(mod_fam) == len(mod_ssca)
        for fam_dx in mod_fam["results"]:
            if fam_dx["kind"] in dx_add:
                mod_ssca["results"].append(fam_dx)
        assert len(mod_ssca["results"]) == len(mod_fam["results"])

    with output_file.open("w") as f:
        print(f"Saving to {output_file.name}")
        json.dump(ssca_results, f)
