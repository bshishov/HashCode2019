import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
from time import sleep
import pickle
import utils

import search.csp as csp
import search.viz as viz

#LEFT = 0
#RIGHT = 1
#UP = 2
#DOWN = 3
#ANY = -1
LEFT = 'left'
RIGHT = 'right'
UP = 'up'
DOWN = 'down'
ANY = 'any'


class Problem(csp.CspProblem):
    def __init__(self, state: np.ndarray, domain, constraints: dict):
        self.state = state
        self.constraints = constraints
        self.domain = domain
        self.rows, self.cols = state.shape

    def iterate_variables(self):
        for i in range(self.rows):
            for j in range(self.cols):
                yield i, j

    def domain_values(self, var) -> list:
        i, j = var
        return self.domain[i, j]

    def remove_from_domain(self, var, val):
        #print('Removing: {} from {}'.format(val, var))
        i, j = var
        self.domain[i, j].remove(val)

    def get_constraints(self):
        pass

    def iterate_neighbours(self, var) -> list:
        i, j = var
        if i > 0:
            yield i-1, j
        if i < self.rows - 1:
            yield i+1, j
        if j > 0:
            yield i, j-1
        if j < self.cols - 1:
            yield i, j+1

    def count_conflicts(self, var, val) -> int:
        i, j = var

        conflicts = 0
        for v, direction in self.constraints[val]:
            if v == ANY:
                continue

            if i > 0:
                if direction == UP and self.state[i - 1, j] != v:
                    conflicts += 1
            if i < self.state.shape[0] - 1:
                if direction == DOWN and self.state[i + 1, j] != v:
                    conflicts += 1
            if j > 0:
                if direction == LEFT and self.state[i, j - 1] != v:
                    conflicts += 1
            if j < self.state.shape[1] - 1:
                if direction == RIGHT and self.state[i, j + 1] != v:
                    conflicts += 1
        return conflicts

    def values_conflicting(self, var1, val1, var2, val2) -> bool:
        i1, j1 = var1
        i2, j2 = var2
        dx = j2 - j1
        dy = i2 - i1
        v_dir = 0
        if dx == 1:
            v_dir = RIGHT
        if dx == -1:
            v_dir = LEFT
        if dy == 1:
            v_dir = DOWN
        if dy == -1:
            v_dir = UP

        # For every constraint for var1
        for c_val, c_dir in self.constraints[val1]:
            # Constrains are defined as VAL, DIRECTION
            # So in order to satisfy constraint the val2 should be equal to constraint val
            if v_dir == c_dir and (val2 == c_val or c_val == ANY):
                # No conflict, at least one constraint satisfied
                return False
        return True

    def get_value(self, var):
        i, j = var
        return self.state[i, j]

    def set_value(self, var, val):
        i, j = var
        self.state[i, j] = val


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


def create_state(rows, cols):
    state = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            state[i, j] = []
    return state


def analyze_shape(pizza: np.ndarray,
                  min_ingredients: int,
                  max_slice: int,
                  offset: int,
                  shape_w: int,
                  shape_h: int):
    rows, cols = pizza.shape
    entropy = np.zeros(pizza.shape, dtype=np.uint8)
    domain = create_state(rows, cols)

    for i in range(rows):
        for j in range(cols):
            r1 = i
            r2 = i + shape_h
            c1 = j
            c2 = j + shape_w
            if is_valid_slice(pizza, r1, r2, c1, c2, min_ingredients, max_slice):
                entropy[r1:r2, c1:c2] += 1

                # Mark all the cells inside slice that they could be potentially sliced by this slice
                for i2 in range(r1, r2):
                    for j2 in range(c1, c2):
                        i_local = i2 - r1
                        j_local = j2 - c1
                        domain[i2, j2].append(offset + i_local * shape_w + j_local)

    # Return all values
    return entropy, domain, offset


def analyze(pizza: np.ndarray, min_ingredients: int, max_slice: int, pool_class=multiprocessing.Pool):
    valid_shapes = get_possible_shapes(min_ingredients, max_slice)
    for shape_idx, shape in enumerate(valid_shapes):
        print('Shape {0}: {1}'.format(shape_idx, shape))

    rows, cols = pizza.shape
    domain = create_state(rows, cols)

    entropy = np.zeros(pizza.shape, dtype=np.uint16)

    pool = pool_class()
    results = []
    offset = 0
    for shape_idx in range(len(valid_shapes)):
        w, h = valid_shapes[shape_idx]
        r = pool.apply_async(analyze_shape, kwds={
            'pizza': pizza,
            'min_ingredients': min_ingredients,
            'max_slice': max_slice,
            'offset': offset,
            'shape_w': w,
            'shape_h': h
        })
        offset += w * h
        results.append(r)

    pool.close()
    pool.join()

    for r in results:
        local_entropy, local_state, local_shape_idx = r.get()
        entropy += local_entropy

        # State aggregation
        for i in range(rows):
            for j in range(cols):
                assert local_entropy[i, j] == len(local_state[i, j])
                domain[i, j] += local_state[i, j]

                """
                for rect_idx in local_state[i, j]:
                    state[i, j].append((rect_idx, local_shape_idx))
                """

    return entropy, domain


def get_index(i, j, cols, offset=0):
    return i * cols + j + offset


def solve(pizza: np.ndarray, min_ingredients: int, max_slice: int):
    rows, cols = pizza.shape
    state = np.empty(pizza.shape, dtype=np.int16)
    entropy, domain = analyze(pizza, min_ingredients, max_slice, ThreadPool)
    shapes = get_possible_shapes(ingredients=min_ingredients, max_slice=max_slice)
    constraints = []

    """ Build constraints """
    """
    offset = 0
    for shape in shapes:
        n = shape[0] * shape[1]
        print(np.reshape(np.arange(n) + offset, shape))
        offset += n
    any_val = list(range(offset))
    """

    offset = 0
    for shape in shapes:
        shape_rows, shape_cols = shape
        for i in range(shape_rows):
            for j in range(shape_cols):
                left = get_index(i, j - 1, cols=shape_cols, offset=offset)
                right = get_index(i, j, cols=shape_cols, offset=offset)
                up = get_index(i - 1, j, cols=shape_cols, offset=offset)
                down = get_index(i, j, cols=shape_cols, offset=offset)

                if j > 0:
                    constraints.append((right, left, LEFT))
                    constraints.append((left, right, RIGHT))

                if i > 0:
                    constraints.append((down, up, UP))
                    constraints.append((up, down, DOWN))

                if i == 0:
                    constraints.append((down, ANY, UP))

                if i == shape_rows - 1:
                    constraints.append((down, ANY, DOWN))

                if j == 0:
                    constraints.append((right, ANY, LEFT))

                if j == shape_cols - 1:
                    constraints.append((right, ANY, RIGHT))

        offset += shape[0] * shape[1]

    constraints_dict = defaultdict(list)
    for src, tgt, direction in constraints:
        constraints_dict[src].append((tgt, direction))

    for k, v in constraints_dict.items():
        print(k, v)

    offset = 0
    for shape in shapes:
        n = shape[0] * shape[1]
        print(np.reshape(np.arange(n) + offset, shape))
        offset += n

    # AC-3 arc consistency (domain reduction)
    #csp.reduce_domain_ac3(domain, constraints_dict)
    problem = Problem(state, domain, constraints_dict)
    csp.arc_consistency3(problem)

    # Random initial state
    for i in range(rows):
        for j in range(cols):
            state[i, j] = np.random.choice(domain[i, j])

    # Count initial conflicts
    conflicts = np.zeros(pizza.shape, dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            conflicts[i, j] = count_conflicts(state[i, j], i, j, state, constraints_dict)

    """ Visualization """
    wnd = viz.Window(1300, 610)
    pizza_drawer = viz.ImgPlotDrawer(viz.Rect(10, 10, 400, 600), 'Pizza')
    state_drawer = viz.ImgPlotDrawer(viz.Rect(420, 10, 400, 600), 'State')
    conflicts_drawer = viz.ImgPlotDrawer(viz.Rect(830, 10, 400, 600), 'Conflicts')
    wnd.add_drawer(state_drawer)
    wnd.add_drawer(pizza_drawer)
    wnd.add_drawer(conflicts_drawer)
    pizza_drawer.set_value(pizza)
    conflicts_drawer.set_value(conflicts)
    state_drawer.set_value(state)
    wnd.draw()

    while True:
        csp.min_conflicts_step(problem)

        for i, j in problem.iterate_variables():
            conflicts[i, j] = problem.count_conflicts((i, j), problem.get_value((i, j)))

        conflicts_drawer.set_value(conflicts)
        state_drawer.set_value(state)
        wnd.draw()

    while np.sum(conflicts) > 0:
        # Randomly chosen variable from the set of conflicted variables
        indices = np.argwhere(conflicts > 0)
        i, j = indices[np.random.choice(len(indices))]

        prev_conf = conflicts[i, j]
        """
        vals = domain[i, j]
        confs = [count_conflicts(v, i, j, state, constraints_dict) for v in vals]
        p = np.exp(1.0 - np.array(confs) / np.max(confs))
        idx = np.random.choice(len(vals), p=p / p.sum())
        if confs[idx] <= prev_conf:
            state[i, j] = vals[idx]
            conflicts[i, j] = confs[idx]
        """

        # TODO: RANDOM OVER MIN CONFLICTS
        #"""
        min_conf_val = None
        min_conf = 99999
        
        for v in domain[i, j]:
            conf = count_conflicts(v, i, j, state, constraints_dict)
            if conf <= min_conf:
                min_conf = conf
                min_conf_val = v

        if min_conf_val is None:
            print('WHOA')
            break

        if min_conf <= prev_conf:
            state[i, j] = min_conf_val
            conflicts[i, j] = min_conf
        #"""

        #"""
        for _i in range(rows):
            for _j in range(cols):
                conflicts[_i, _j] = count_conflicts(state[_i, _j], _i, _j, state, constraints_dict)
        #"""

        """
        if i > 0:
            conflicts[i-1, j] = count_conflicts(state[i-1, j], i-1, j, state, constraints_dict)
        if i < rows - 1:
            conflicts[i+1, j] = count_conflicts(state[i+1, j], i+1, j, state, constraints_dict)

        if j > 0:
            conflicts[i, j-1] = count_conflicts(state[i, j-1], i, j-1, state, constraints_dict)
        if j < cols - 1:
            conflicts[i, j+1] = count_conflicts(state[i, j+1], i, j+1, state, constraints_dict)
        """

        """
        if i > 0 and is_conflicting(state[i - 1, j], i-1, j, state, constraints_dict):
            conflicting[i - 1, j] = 1

        if i < rows - 1 and is_conflicting(state[i + 1, j], i+1, j, state, constraints_dict):
            conflicting[i + 1, j] = 1

        if j > 0 and is_conflicting(state[i, j - 1], i, j - 1, state, constraints_dict):
            conflicting[i, j - 1] = 1

        if j > cols - 1 and is_conflicting(state[i, j - 1], i, j + 1, state, constraints_dict):
            conflicting[i, j + 1] = 1
        """

        state_drawer.set_value(state)
        conflicts_drawer.set_value(conflicts)
        wnd.draw()

    print('DONW')
    print(state)


def is_conflicting(val, i, j, state, constraints):
    for val, direction in constraints[val]:
        if i > 0:
            if direction == UP and state[i - 1, j] != val:
                return True
        if i < state.shape[0] - 1:
            if direction == DOWN and state[i + 1, j] != val:
                return True
        if j > 0:
            if direction == LEFT and state[i, j - 1] != val:
                return True
        if j < state.shape[1] - 1:
            if direction == RIGHT and state[i, j + 1] != val:
                return True
    return False


def count_conflicts(val, i, j, state, constraints):
    conflicts = 0
    for v, direction in constraints[val]:
        if i > 0:
            if direction == UP and state[i - 1, j] != v:
                conflicts += 1
        if i < state.shape[0] - 1:
            if direction == DOWN and state[i + 1, j] != v:
                conflicts += 1
        if j > 0:
            if direction == LEFT and state[i, j - 1] != v:
                conflicts += 1
        if j < state.shape[1] - 1:
            if direction == RIGHT and state[i, j + 1] != v:
                conflicts += 1
    return conflicts