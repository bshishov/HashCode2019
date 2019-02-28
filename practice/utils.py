import numpy as np


def multi_iterator(*iterables):
    iterables = [iter(i) for i in iterables]

    while len(iterables) > 0:
        for iterable in iterables:
            try:
                yield next(iterable)
            except StopIteration:
                iterables.remove(iterable)


def slice_array(pizza: np.ndarray, axis: int, pivot: int):
    """
    :param pizza: The pizza array
    :param axis: 0 - horizontal slice, 1 - vertical slice
    :param pivot: location to slice
    :return: arrays slice1, slice2 obtained by slicing
    """
    if axis == 0:
        return pizza[:pivot, :], pizza[pivot:, :]

    return pizza[:, :pivot], pizza[:, pivot:]


def get_possible_dimensions(size: int) -> list:
    dimensions = []
    for i in range(1, size + 1):
        if size % i == 0:
            dimensions.append((i, size // i))
    return dimensions


def get_possible_shapes(ingredients: int, max_slice: int) -> list:
    valid_shapes = []
    min_size = ingredients * 2
    for s in range(min_size, max_slice + 1):
        for d in get_possible_dimensions(s):
            valid_shapes.append(d)
    return valid_shapes


def is_valid_slice(pizza, r1, r2, c1, c2, min_ingredients: int, max_slice: int):
    rows = r2 - r1
    cols = c2 - c1
    size = rows * cols

    if c2 > pizza.shape[1] or r2 > pizza.shape[0]:
        # out of bounds
        return False

    if size > max_slice:
        return False

    oversize = size - 2 * min_ingredients

    # If the slice is too small
    if oversize < 0:
        return False

    #s = pizza[r1:r2, c1:c2]

    tomatoes = 0
    mushrooms = 0
    for i in range(r1, r2):
        for j in range(c1, c2):
            ing = pizza[i, j]
            if ing == 1:
                tomatoes += 1
            if ing == -1:
                mushrooms += 1

    if tomatoes < min_ingredients:
        return False

    if mushrooms < min_ingredients:
        return False


    """
    if np.count_nonzero(s == 1) < min_ingredients:
        return False

    if np.count_nonzero(s == -1) < min_ingredients:
        return False
    """

    # return np.abs(pizza[r1:r2, c1:c2].sum()) <= oversize
    return True