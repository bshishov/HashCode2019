import numpy as np
import multiprocessing
import multiprocessing.dummy
from collections import namedtuple

TOMATO = 1
LEFT = 'left'
RIGHT = 'right'
UP = 'up'
DOWN = 'down'
ANY = 'any'
SOLUTION_DTYPE = np.int16
NO_SOLUTION_1X1 = np.array([[-1]], dtype=SOLUTION_DTYPE)

Constraints = namedtuple('Constraints', ['lr_pairs', 'l_constrained', 'r_constrained',
                                         'ud_pairs', 'u_constrained', 'd_constrained'])


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


def is_valid_slice(pizza: np.ndarray, r1, r2, c1, c2, min_ingredients: int, max_slice: int) -> bool:
    rows = r2 - r1
    cols = c2 - c1
    size = rows * cols

    if c2 > pizza.shape[1] or r2 > pizza.shape[0]:
        # out of bounds
        return False

    if size > max_slice:
        return False

    # If the slice is too small
    if size < 2 * min_ingredients:
        return False

    tomatoes = 0
    mushrooms = 0
    for i in range(r1, r2):
        for j in range(c1, c2):
            ing = pizza[i, j]
            if ing == TOMATO:
                tomatoes += 1
            else:
                mushrooms += 1

    if tomatoes < min_ingredients:
        return False

    if mushrooms < min_ingredients:
        return False

    return True


def create_domain(rows, cols):
    state = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            state[i, j] = []
    return state


def generate_domain_for_single_shape(pizza: np.ndarray,
                                     min_ingredients: int,
                                     max_slice: int,
                                     offset: int,
                                     shape_cols: int,
                                     shape_rows: int):
    rows, cols = pizza.shape
    domain = create_domain(rows, cols)

    for i in range(rows):
        for j in range(cols):
            r1 = i
            r2 = i + shape_rows
            c1 = j
            c2 = j + shape_cols
            if is_valid_slice(pizza, r1, r2, c1, c2,
                              min_ingredients=min_ingredients,
                              max_slice=max_slice):
                # Mark all the cells inside slice that they could be potentially sliced by this slice
                for i2 in range(r1, r2):
                    for j2 in range(c1, c2):
                        i_local = i2 - r1
                        j_local = j2 - c1
                        domain[i2, j2].append(offset + i_local * shape_cols + j_local)

    # Return all values
    return domain


def generate_domain_parallel(pizza: np.ndarray,
                             min_ingredients: int,
                             max_slice: int,
                             valid_shapes: list,
                             pool_class=multiprocessing.Pool):
    for shape_idx, shape in enumerate(valid_shapes):
        print('[Solver] Shape {0}: {1}'.format(shape_idx, shape))

    rows, cols = pizza.shape
    domain = create_domain(rows, cols)

    pool = pool_class()
    results = []
    offset = 0
    for shape_rows, shape_cols in valid_shapes:
        r = pool.apply_async(generate_domain_for_single_shape, kwds={
            'pizza': pizza,
            'min_ingredients': min_ingredients,
            'max_slice': max_slice,
            'offset': offset,
            'shape_rows': shape_rows,
            'shape_cols': shape_cols
        })
        offset += shape_rows * shape_cols
        results.append(r)

    pool.close()
    pool.join()

    for r in results:
        shape_domain = r.get()

        # Merge shape domains
        for i in range(rows):
            for j in range(cols):
                domain[i, j] += shape_domain[i, j]
    return domain


def generate_domain_simple(pizza: np.ndarray,
                           min_ingredients: int,
                           max_slice: int,
                           valid_shapes: list):
    rows, cols = pizza.shape
    domain = create_domain(rows, cols)

    offset = 0
    for shape_rows, shape_cols in valid_shapes:
        for i in range(rows):
            for j in range(cols):
                r1 = i
                r2 = i + shape_rows
                c1 = j
                c2 = j + shape_cols
                if is_valid_slice(pizza, r1, r2, c1, c2, min_ingredients, max_slice):
                    # Mark all the cells inside slice that they could be potentially sliced by this slice
                    for i2 in range(r1, r2):
                        for j2 in range(c1, c2):
                            i_local = i2 - r1
                            j_local = j2 - c1
                            domain[i2, j2].append(offset + i_local * shape_cols + j_local)
        offset += shape_cols * shape_rows

    # Return all values
    return domain


def get_index(i, j, cols, offset=0):
    return i * cols + j + offset


def neighbors(i, j, rows, cols):
    n = []
    if i > 0:
        n.append((i - 1, j, i, j))
    if i < rows - 1:
        n.append((i + 1, j, i, j))
    if j > 0:
        n.append((i, j - 1, i, j))
    if j < cols - 1:
        n.append((i, j + 1, i, j))
    return n


def reduce_domain_ac3(domain: np.ndarray, constraints: Constraints, verbose=False):
    rows, cols = domain.shape
    q = []

    it = np.nditer(domain, flags=['multi_index', 'refs_ok'])
    while not it.finished:
        i, j = it.multi_index

        consistent = True
        if i > 0 and remove_inconsistent(i, j, i - 1, j, domain, constraints, verbose):
            consistent = False
        if i < rows - 1 and remove_inconsistent(i, j, i + 1, j, domain, constraints, verbose):
            consistent = False
        if j > 0 and remove_inconsistent(i, j, i, j - 1, domain, constraints, verbose):
            consistent = False
        if j < cols - 1 and remove_inconsistent(i, j, i, j + 1, domain, constraints, verbose):
            consistent = False

        if not consistent:
            q += neighbors(i, j, rows, cols)

        it.iternext()

    while len(q) > 0:
        i_src, j_src, i_tgt, j_tgt = q.pop(0)
        if remove_inconsistent(i_src, j_src, i_tgt, j_tgt, domain, constraints, verbose):
            q += neighbors(i_src, j_src, rows, cols)


def remove_inconsistent(i_src, j_src, i_tgt, j_tgt, domain: np.ndarray, constraints: Constraints, verbose=False):
    removed = False
    # For value a in arc
    for v_src in domain[i_src, j_src]:
        can_satisfy_constraint = False

        # For value b in arc
        for v_tgt in domain[i_tgt, j_tgt]:
            x_dir = j_tgt - j_src
            y_dir = i_tgt - i_src

            valid_assignment = True
            if x_dir == -1:
                # v_tgt - left, v_src - right
                if v_tgt in constraints.r_constrained or v_src in constraints.l_constrained:
                    valid_assignment = (v_tgt, v_src) in constraints.lr_pairs
            elif x_dir == 1:
                # v_tgt - right, v_src - left
                if v_src in constraints.r_constrained or v_tgt in constraints.l_constrained:
                    valid_assignment = (v_src, v_tgt) in constraints.lr_pairs
            elif y_dir == -1:
                # v_tgt - up, v_src - down
                if v_tgt in constraints.d_constrained or v_src in constraints.u_constrained:
                    valid_assignment = (v_tgt, v_src) in constraints.ud_pairs
            elif y_dir == 1:
                # v_tgt - down, v_src - up
                if v_src in constraints.d_constrained or v_tgt in constraints.u_constrained:
                    valid_assignment = (v_src, v_tgt) in constraints.ud_pairs

            if valid_assignment:
                can_satisfy_constraint = True
                break

        if not can_satisfy_constraint:
            if verbose:
                print('[AC3] removed val {0} from ({1}, {2})'.format(v_src, i_src, j_src))
            domain[i_src, j_src].remove(v_src)
            removed = True

    return removed


def find_constraints(shapes: list) -> Constraints:
    constraints = Constraints(l_constrained=set(),
                              r_constrained=set(),
                              u_constrained=set(),
                              d_constrained=set(),
                              lr_pairs=set(),
                              ud_pairs=set())

    offset = 0
    for shape_rows, shape_cols in shapes:
        for i in range(shape_rows):
            for j in range(shape_cols):
                left = get_index(i, j - 1, cols=shape_cols, offset=offset)
                right = get_index(i, j, cols=shape_cols, offset=offset)
                up = get_index(i - 1, j, cols=shape_cols, offset=offset)
                down = get_index(i, j, cols=shape_cols, offset=offset)

                if j > 0:
                    constraints.lr_pairs.add((left, right))
                    constraints.l_constrained.add(right)  # has requirement of the left
                    constraints.r_constrained.add(left)  # has requirement of the right

                if i > 0:
                    constraints.ud_pairs.add((up, down))
                    constraints.u_constrained.add(down)  # has requirement of the top
                    constraints.d_constrained.add(up)  # has requirement of the bottom

        offset += shape_rows * shape_cols
    return constraints


def split(arr: np.ndarray, axis: int, delimiter=None):
    if delimiter is None:
        delimiter = arr.shape[axis] // 2
        # Golden ratio, second part will be short
        #delimiter = int(arr.shape[axis] * 0.61803399)

    if axis == 0:
        return arr[:delimiter, :], arr[delimiter:, :]
    else:
        return arr[:, :delimiter], arr[:, delimiter:]


def merge(x: np.ndarray, y: np.ndarray, axis: int):
    assert x.shape[axis - 1] == y.shape[axis - 1]
    return np.concatenate((x, y), axis=axis)


def is_valid_assignment_top_bottom(top_val: int, bottom_val: int, requirements):
    if top_val in requirements.d_constrained or bottom_val in requirements.u_constrained:
        return (top_val, bottom_val) in requirements.ud_pairs
    return True


def is_valid_assignment_left_right(left_val: int, right_val: int, requirements):
    if left_val in requirements.r_constrained or right_val in requirements.l_constrained:
        return (left_val, right_val) in requirements.lr_pairs
    return True


def check_if_can_be_merged(x: np.ndarray, y: np.ndarray, requirements, axis: int) -> bool:
    if axis == 0:
        assert x.shape[1] == y.shape[1]
        for i in range(x.shape[1]):
            if not is_valid_assignment_top_bottom(x[-1, i], y[0, i], requirements):
                return False
        return True
    else:
        assert x.shape[0] == y.shape[0]
        for i in range(x.shape[0]):
            if not is_valid_assignment_left_right(x[i, -1], y[i, 0], requirements):
                return False
        return True


def iterate_solutions(domain: np.ndarray, requirements: Constraints) -> list:
    """ Solves the small chunk and returns the list of possible states """
    rows, cols = domain.shape

    if rows == 1 and cols == 1:
        local_solutions = 0
        for possible_value in domain[0, 0]:
            local_solutions += 1
            yield np.array([[possible_value]], dtype=SOLUTION_DTYPE)
        if local_solutions == 0:
            # No solutions found return a -1
            yield NO_SOLUTION_1X1
    else:
        if rows > cols:
            axis = 0
        else:
            axis = 1

        x_domain, y_domain = split(domain, axis=axis)
        y_cache = []  # to remove second reiteration
        local_solutions = 0

        for i, x in enumerate(iterate_solutions(x_domain, requirements)):
            if i == 0:
                for y in iterate_solutions(y_domain, requirements):
                    y_cache.append(y)
                    if check_if_can_be_merged(x, y, requirements, axis=axis):
                        local_solutions += 1
                        yield merge(x, y, axis=axis)
            else:
                for y in y_cache:
                    if check_if_can_be_merged(x, y, requirements, axis=axis):
                        local_solutions += 1
                        yield merge(x, y, axis=axis)

        if local_solutions == 0:
            # No solutions found return a patch of "-1"s
            yield np.full(domain.shape, dtype=SOLUTION_DTYPE, fill_value=-1)


def generate_shape_to_slice_map(shapes):
    s2s_map = {}
    offset = 0
    for shape_rows, shape_cols in shapes:
        s2s_map[offset] = (shape_rows, shape_cols)
        offset += shape_rows * shape_cols
    return s2s_map


class Solver(object):
    def __init__(self, min_ingredients: int, max_slice_size: int):
        self.min_ingredients = min_ingredients
        self.max_slice_size = max_slice_size
        self.slice_shapes = get_possible_shapes(self.min_ingredients, self.max_slice_size)
        self.constraints = find_constraints(self.slice_shapes)
        self.shape_to_slice_map = {}
        self.slice_mapping = generate_shape_to_slice_map(self.slice_shapes)

    def explain_shapes(self):
        offset = 0
        for index, shape in enumerate(self.slice_shapes):
            shape_rows, shape_cols = shape
            print('[Solver] Slice shape {0}  rows={1} cols={2}'.format(index, shape[0], shape[1]))
            for i in range(shape_rows):
                print('\t'.join([str(get_index(i, j, cols=shape_cols, offset=offset)) for j in range(shape_cols)]))
            offset += shape_rows * shape_cols

    def explain_constraints(self):
        print('[Solver] Left-Right pairs')
        print(self.constraints.lr_pairs)

        print('[Solver] Up-Down pairs')
        print(self.constraints.ud_pairs)

    def iterate_solutions(self,
                          pizza: np.ndarray,
                          domain_creation='simple',
                          ac3_reduction=False,
                          verbose=False):
        if verbose:
            print('[Solver] Generating domain with strategy: {0}'.format(domain_creation))
        if domain_creation is 'simple':
            domain = generate_domain_simple(pizza,
                                            min_ingredients=self.min_ingredients,
                                            max_slice=self.max_slice_size,
                                            valid_shapes=self.slice_shapes)
        elif domain_creation == 'multiprocess':
            domain = generate_domain_parallel(pizza,
                                              min_ingredients=self.min_ingredients,
                                              max_slice=self.max_slice_size,
                                              valid_shapes=self.slice_shapes,
                                              pool_class=multiprocessing.Pool)
        elif domain_creation == 'multithread':
            domain = generate_domain_parallel(pizza,
                                              min_ingredients=self.min_ingredients,
                                              max_slice=self.max_slice_size,
                                              valid_shapes=self.slice_shapes,
                                              pool_class=multiprocessing.dummy.Pool)
        else:
            raise RuntimeError('Invalid argument: domain_creation')

        if ac3_reduction:
            if verbose:
                print('[Solver] Performing AC3 reduction')
            reduce_domain_ac3(domain, self.constraints, verbose=verbose)

        if verbose:
            print('[Solver] Solving...')
        for solution in iterate_solutions(domain=domain.view(), requirements=self.constraints):
            yield solution

    def first_solution(self, *args, **kwargs):
        for solution in self.iterate_solutions(*args, **kwargs):
            return solution

    def iterate_slices(self, solution: np.ndarray):
        rows, cols = solution.shape
        for i in range(rows):
            for j in range(cols):
                slice_shape = self.slice_mapping.get(solution[i, j], None)
                if slice_shape is not None:
                    yield i, i + slice_shape[0], j, j + slice_shape[1]
