# /usr/bin/env python

import numpy as np
from typing import List, Tuple


def is_golay_pair(a: np.array, b: np.array):
    # sum = 0
    for k in range(len(a)):
        ...
        # for j in range()


def ls_code() -> np.array:
    L0 = 4
    # k = 1
    # Np = 8

    M = 2**3
    Lc = M * L0
    Z = Lc // M

    p = +1
    m = -1
    c1 = np.array([p, p, m, p], dtype=np.int8)
    s1 = np.array([p, m, m, m], dtype=np.int8)
    c2 = np.array([p, p, p, m], dtype=np.int8)
    s2 = np.array([p, m, p, p], dtype=np.int8)

    def recurse(
        c1: List[int], s1: List[int], c2: List[int], s2: List[int]
    ) -> Tuple[List[np.array], List[np.array]]:
        p1 = [
            c1 * c2,
            s1 * s2,
            c1 - c2,
            s1 - s2,
        ]
        p2 = [
            c2 * c1,
            s2 * s1,
            c2 - c1,
            s2 - s1,
        ]
        return p1, p2

    l1: List[np.array] = [c1, s1, c2, s2]
    l2a, l2b = recurse(*l1)
    l3a, l3b = recurse(*l2a)
    l3c, l3d = recurse(*l2b)

    l_codes = [l1, l2a, l2b, l3a, l3b, l3c, l3d]

    ls_sequence = np.ravel(
        [np.concatenate((ls[0], np.zeros(Z - 1), ls[1])) for ls in l_codes]
    )

    return ls_sequence


if __name__ == "__main__":
    ls_code()
