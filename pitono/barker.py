#!/usr/bin/env python3

from typing import List


def get_barker(len: int) -> List[bool]:
    codes = {
        # fmt: off
        2: [True, False],  # [True, True],
        3: [True, True, False],
        4: [True, True, False, True, True, True, True, False, False],
        5: [True, True, True, False, True, False],
        7: [True, True, True, False, False, True, False, False],
        11: [True, True, True, False, False, False, True, False, False, True, False],
        13: [True, True, True, True, True, False, False, True, True, False, True, False, True],
        # fmt: on
    }
    if len not in codes.keys:
        raise Exception
    else:
        return codes[len]
