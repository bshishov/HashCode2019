import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
from time import sleep
import pickle

import search.viz as viz


def is_valid_slice(pizza, r1, r2, c1, c2, min_ingredients: int, max_slice: int):
    """ Checks whether the slice (r1, c1, r2, c2) is valid in terms of size and content
        according to the original problem """
    rows = r2 - r1
    cols = c2 - c1

    # Bounds check
    if c2 > pizza.shape[1] or r2 > pizza.shape[0]:
        return False

    # Size check
    size = rows * cols
    if size > max_slice:
        return False

    oversize = size - 2 * min_ingredients

    # If the slice is too small
    if oversize < 0:
        return False

    # Content check
    content = pizza[r1:r2, c1:c2]

    # Mushrooms
    mushrooms = np.count_nonzero(content == 1)
    if mushrooms < min_ingredients:
        return False

    # Tomatoes - are all remaining cells
    if size - mushrooms < min_ingredients:
        return False

    return True


def create_2d_array_of_lists(rows, cols):
    """ Constructs 2D numpy object array of specified size
        and fills each cell with a new instance of list """
    state = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            state[i, j] = list()
    return state


def iterate_divisors(size):
    """ Iterates over all possible divisors and dividers of the input

        Example:
            iterate_divisors(size=6)
            output (iterable): (1, 6), (2, 3), (3, 2), (6, 1)
    """
    for i in range(1, size + 1):
        if size % i == 0:
            yield i, size // i


def get_possible_shapes(min_ingredients: int, slice_cap: int):
    """ Returns the list of all possible shape sizes according to the original problem """
    valid_shapes = []
    min_size = min_ingredients * 2
    max_size = slice_cap
    for size in range(min_size, max_size + 1):
        for dimensions in iterate_divisors(size):
            valid_shapes.append(dimensions)
    return valid_shapes


def analyze_shape(pizza: np.ndarray,
                  min_ingredients: int,
                  max_slice: int,
                  shape_id: int,
                  shape_w: int,
                  shape_h: int):
    """ Pizza content analysis be iterating over each cell and checking
        whether or not a slice of specified shape could be built from this cell.

        Pseudo code:
        for (i, j):
            If it is possible to do a valid slice in (i, j) then:
                for each cell that is covered by slice (i2, j2):
                    increase entropy by 1: entropy(i2, j2) += 1
                    add current slice info (i, j, shape_id) to state of cell (i2, j2)
        return entropy, state

        :returns
            entropy - 2d array of uint8, number of potential slices (of specified shape) for each cell.
            state - 2d array of lists, list of all possible slices (of specified shape) for each cell.
    """
    rows, cols = pizza.shape
    entropy = np.zeros(pizza.shape, dtype=np.uint8)
    state = create_2d_array_of_lists(rows, cols)

    for i in range(rows):
        for j in range(cols):
            r1 = i
            r2 = i + shape_h
            c1 = j
            c2 = j + shape_w
            if is_valid_slice(pizza, r1, r2, c1, c2, min_ingredients, max_slice):
                # Index of top-left corner of a slice
                entropy[r1:r2, c1:c2] += 1

                # Mark all the cells inside slice that they could be potentially sliced by this slice
                for i2 in range(r1, r2):
                    for j2 in range(c1, c2):
                        state[i2, j2].append((i, j, shape_id))

    # Return all values
    return entropy, state


def analyze(pizza: np.ndarray, min_ingredients: int, max_slice: int, pool_class=multiprocessing.Pool):
    """ Calculates all possible slice variations for each cell of the pizza.

        Calculations are performed in a parallel way with process pool:
            One process for each shape

        :returns
            entropy - 2d array of uint16, number of potential slices for each cell.
            state - 2d array of lists, list of all possible slices for each cell.
    """
    valid_shapes = get_possible_shapes(min_ingredients, max_slice)
    for shape_id, shape in enumerate(valid_shapes):
        print('Shape {0}: {1}'.format(shape_id, shape))

    rows, cols = pizza.shape
    state = create_2d_array_of_lists(rows, cols)

    entropy = np.zeros(pizza.shape, dtype=np.uint16)

    pool = pool_class()
    results = []
    for shape_id in range(len(valid_shapes)):
        w, h = valid_shapes[shape_id]
        r = pool.apply_async(analyze_shape, kwds={
            'pizza': pizza,
            'min_ingredients': min_ingredients,
            'max_slice': max_slice,
            'shape_id': shape_id,
            'shape_w': w,
            'shape_h': h
        })
        results.append(r)

    pool.close()
    pool.join()

    """ Results aggregation """
    for r in results:
        local_entropy, local_state = r.get()
        entropy += local_entropy

        # State aggregation
        for i in range(rows):
            for j in range(cols):
                assert local_entropy[i, j] == len(local_state[i, j])
                state[i, j] += local_state[i, j]
    return entropy, state


class Commit(object):
    def __init__(self, t, action):
        self.t = t
        self.truncated = []
        self.action = action
        self.blacklist = set()
        self.reverted = False

    def add_truncation(self, i: int, j: int, state: tuple):
        self.truncated.append((i, j, state))

    def apply(self, mask: np.ndarray, state: np.ndarray, entropy: np.ndarray, valid_shapes: list):
        slice_r1, slice_c1, slice_shape_idx = self.action
        slice_w, slice_h = valid_shapes[slice_shape_idx]
        slice_r2 = slice_r1 + slice_h
        slice_c2 = slice_c1 + slice_w

        mask[slice_r1:slice_r2, slice_c1:slice_c2] = 1
        for i, j, s in self.truncated:
            state[i, j].remove(s)
            entropy[i, j] -= 1

    def revert(self, mask: np.ndarray, state: np.ndarray, entropy: np.ndarray, valid_shapes: list):
        if self.reverted:
            return
        self.reverted = True

        self.blacklist.add(self.action)

        slice_r1, slice_c1, slice_shape_idx = self.action
        slice_w, slice_h = valid_shapes[slice_shape_idx]
        slice_r2 = slice_r1 + slice_h
        slice_c2 = slice_c1 + slice_w

        mask[slice_r1:slice_r2, slice_c1:slice_c2] = 0
        for i, j, s in self.truncated:
            state[i, j].append(s)
            entropy[i, j] += 1


def solve(pizza: np.ndarray, min_ingredients: int, max_slice: int):
    """ Visualization """
    wnd = viz.Window(1300, 610)
    pizza_drawer = viz.ImgPlotDrawer(viz.Rect(10, 10, 400, 600), 'Pizza')
    mask_drawer = viz.ImgPlotDrawer(viz.Rect(420, 10, 400, 600), 'Mask')
    entropy_drawer = viz.ImgPlotDrawer(viz.Rect(830, 10, 400, 600), 'Entropy')
    wnd.add_drawer(mask_drawer)
    wnd.add_drawer(pizza_drawer)
    wnd.add_drawer(entropy_drawer)
    pizza_drawer.set_value(pizza)

    # Multiprocessing
    #entropy, state = analyze(pizza, min_ingredients, max_slice)

    # Multithreading
    entropy, state = analyze(pizza, min_ingredients, max_slice, pool_class=ThreadPool)

    valid_shapes = get_possible_shapes(min_ingredients, max_slice)

    # Set mask to 0 where entropy is nonzero
    mask = np.ones(pizza.shape, dtype=np.uint8)
    mask[np.nonzero(entropy)] = 0

    rows, cols = pizza.shape

    for i in range(rows):
        for j in range(cols):
            # Just to make sure we are doing everything right
            assert entropy[i, j] == len(state[i, j])
            #entropy[i, j] += ((i - rows // 2) ** 2 + (j - cols // 2) ** 2)

    print('Total values: {0} Dead cells: {1}'.format(entropy.size, np.sum(mask)))
    mask_drawer.set_value(mask)
    entropy_drawer.set_value(entropy)
    cell_affected_by_commits = create_2d_array_of_lists(rows, cols)

    iteration = 0
    commits = []
    banned = set()

    while np.sum(mask) < pizza.size:
        """ Step 1. Cell selection """
        i, j = np.unravel_index(np.argmin(entropy[np.nonzero(entropy)]), entropy.shape)

        valid_actions = list(set(state[i, j]) - banned)

        """ If current cell is in the conflict state (no possible options) """
        if len(valid_actions) == 0:
            cell_commits = cell_affected_by_commits[i, j]
            if len(cell_commits) > 0:
                print('Reverting: {}, {}'.format(i, j))
                c = cell_commits.pop()
                banned.add(c.action)
                c.revert(mask, state, entropy, valid_shapes)
                continue
            break

            if len(commits) == 0:
                print('Nothing to revert break')
                break

            commit = commits.pop()
            commit.revert(mask, state, entropy, valid_shapes)

            mask_drawer.set_value(mask)
            entropy_drawer.set_value(entropy)
            wnd.draw()
            #sleep(1)
            continue

        action = valid_actions[np.random.randint(len(valid_actions))]
        commit = Commit(iteration, action)

        # Slice / do action
        slice_r1, slice_c1, slice_shape_idx = action
        slice_w, slice_h = valid_shapes[slice_shape_idx]
        slice_r2 = slice_r1 + slice_h
        slice_c2 = slice_c1 + slice_w

        """
        print('\tIndex: ({0}, {1})'.format(i, j))
        print('\tState: {0}'.format(s))
        print('\tChoice: {0} '.format(choice))
        print('\tSlice ', slice_r1, slice_c1, slice_r2, slice_c2)
        """

        # Update states
        # All states inside the slice are invalid now
        # So we need to truncate states of affected cells
        states_to_truncate = set()
        for i in range(slice_r1, slice_r2):
            for j in range(slice_c1, slice_c2):
                states_to_truncate.update(state[i, j])

        #print('\tStates to truncate: {0}'.format(states_to_truncate))

        # Truncate states in some area
        context_r1 = max(0, slice_r1 - 14)
        context_r2 = min(slice_r2 + 14, rows)
        context_c1 = max(0, slice_c1 - 14)
        context_c2 = min(slice_c2 + 14, cols)

        for valid_actions in states_to_truncate:
            for i in range(context_r1, context_r2):
                for j in range(context_c1, context_c2):
                    if valid_actions in state[i, j]:
                        cell_affected_by_commits[i, j].append(commit)
                        commit.add_truncation(i, j, valid_actions)

        commit.apply(mask, state, entropy, valid_shapes)
        commits.append(commit)

        if iteration % 1 == 0:
            mask_drawer.set_value(mask)
            entropy_drawer.set_value(entropy)
            wnd.draw()

        iteration += 1

    sleep(10)
    return []

