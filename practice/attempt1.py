import sys
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from functools import lru_cache
import typing
import numba

TOMATO = -1
MUSHROOM = 1
global_pizza = None
valid_shapes = []

nb_indices = numba.int32[:]


@numba.jitclass([
    ('r1', numba.int32),
    ('r2', numba.int32),
    ('c1', numba.int32),
    ('c2', numba.int32)
])
class Indices(object):
    def __init__(self, r1, c1, r2, c2):
        self.r1 = r1
        self.r2 = r2
        self.c1 = c1
        self.c2 = c2

    def cols(self):
        return self.c2 - self.c1

    def rows(self):
        return self.r2 - self.r1

    def size(self):
        return (self.c2 - self.c1) * (self.r2 - self.r1)

    def __repr__(self):
        return '<Indices {} {} {} {}>'.format(self.r1, self.r2, self.c1, self.c2)


@numba.jit(nopython=True)
def slice_indices(indices: Indices, axis: int, pivot: int) -> (Indices, Indices):
    if axis == 0:
        return Indices(indices.r1, indices.c1, pivot, indices.r2), \
               Indices(pivot, indices.c1, indices.c2, indices.r2)

    return Indices(indices.r1, indices.c1, indices.r2, pivot), \
           Indices(indices.r1, pivot, indices.r2, indices.c2)


@numba.jit(nopython=True)
def get_slice(pizza: np.ndarray, indices: Indices) -> np.ndarray:
    return pizza[indices.r1:indices.r2, indices.c1:indices.c2]


def get_successors_actions(indices: tuple):
    """
    p = get_slice(global_pizza, indices)

    # exlude potential bad slices?
    if len(p[p > 0]) < min_ingredients:
        return

    if len(p[p < 0]) < min_ingredients:
        return
        """

    # TODO: WISE ENUMERATION!
    # like start from center and exclude bad slices
    # depending on min_ingredients and slice_cap
    # calc min slice size or smth
    """ Returns axis and pivot """
    rows, cols = indices[2] - indices[0], indices[3] - indices[1]

    center_x = cols // 2
    center_y = rows // 2

    horizontal = range(1, rows)
    vertical = range(1, cols)
    # horizontal = sorted(range(1, rows), key=lambda y: abs(y - center_y))
    # vertical = sorted(range(1, cols), key=lambda x: abs(x - center_x))

    horizontal = zip([0] * (rows - 1), horizontal)
    vertical = zip([1] * (cols - 1), vertical)

    for v in multi_iterator(horizontal, vertical):
        yield v


@numba.jit(nopython=True)
def merge_to(target: list, source: list):
    m = len(source)
    i = 0
    while i < m:
        target.insert(m - i - 1, source[m - i - 1])
        i += 1


@numba.jit(nopython=True)
def merge_to_reversed(target: list, source: list):
    m = len(source)
    i = 0
    while i < m:
        target.insert(m - i - 1, source[i])
        i += 1


@numba.jit(nopython=True)
def get_successors_actions_no_iter(indices: Indices) -> list:
    rows = indices.rows()
    cols = indices.cols()
    center_col = cols // 2
    center_row = rows // 2

    c1 = list(zip([1] * (cols - center_col), range(center_col, cols)))
    c2 = list(zip([1] * center_col, range(1, center_col)))

    r1 = list(zip([0] * (rows - center_row), range(center_row, rows)))
    r2 = list(zip([0] * center_row, range(1, center_row)))

    actions = c1
    merge_to_reversed(actions, c2)
    merge_to(actions, r1)
    merge_to_reversed(actions, r2)

    return actions


@numba.jit(nopython=True)
def valid_slice(indices: Indices, min_ingredients: int, slice_cap: int) -> bool:
    # w, h = indices.cols(), indices.rows()
    size = indices.size()

    """
    if (w, h) not in valid_shapes:
        return False        
    """

    """
    # Size of the slice
    size = (indices[2] - indices[0]) * (indices[3] - indices[1])


    """
    if size > slice_cap:
        return False

    oversize = size - (min_ingredients + min_ingredients)

    if oversize < 0:
        return False

    # TODO: verify this magic
    # tomatoes = -1
    # mushrooms = +1
    # so the sum gives us the BALANCE between T and M
    # if BALANCE exceeds oversize then there is more ingredients of some type than needed
    return np.abs(get_slice(global_pizza, indices).sum()) <= oversize

    # Actual fair check
    # return len(pizza[pizza > 0]) >= min_ingredients and len(pizza[pizza < 0]) >= min_ingredients


# @lru_cache()
@numba.jit(nopython=True, parallel=True)
def _solve(indices: Indices, min_ingredients: int, slice_cap: int) -> typing.List[Indices]:
    if indices.size() < min_ingredients + min_ingredients:
        return None

    for axis, pivot in get_successors_actions_no_iter(indices):
        slice1, slice2 = slice_indices(indices, axis=axis, pivot=pivot)
        s1 = _solve(slice1, min_ingredients, slice_cap)
        if s1 is None:
            continue
        s2 = _solve(slice2, min_ingredients, slice_cap)
        if s2 is None:
            continue
        return s1 + s2

    if valid_slice(indices, min_ingredients, slice_cap):
        return [indices]
    return None


def solve(pizza: np.ndarray, min_ingredients: int, max_slice: int):
    global_pizza = pizza
    indices = Indices(0, 0, pizza.shape[0], pizza.shape[1])
    return _solve(indices, min_ingredients, max_slice)
