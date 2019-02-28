import numpy as np
import domain as dm
from collections import defaultdict, namedtuple
import math
from functools import lru_cache
import multiprocessing
import viz


LEFT = 'left'
RIGHT = 'right'
UP = 'up'
DOWN = 'down'
ANY = 'any'
SOLUTION_DTYPE = np.int16
NO_SOLUTION_1X1 = np.array([[-1]], dtype=SOLUTION_DTYPE)

Constraints = namedtuple('Constraints', ['lr_pairs', 'l_constrained', 'r_constrained',
                                         'ud_pairs', 'u_constrained', 'd_constrained'])


def hash_mask(arr: np.ndarray, mask: np.ndarray):
    return hash(tuple(arr.flat[np.nonzero(mask.flatten())]))


def split(arr: np.ndarray, axis: int, delimiter=None):
    if delimiter is None:
        #delimiter = arr.shape[axis] // 2
        # Golden ratio, second part will be short
        delimiter = int(arr.shape[axis] * 0.61803399)

    if axis == 0:
        return arr[:delimiter, :], arr[delimiter:, :]
    else:
        return arr[:, :delimiter], arr[:, delimiter:]


def split_fill_mask(arr: np.ndarray, axis: int, delimiter=None):
    if axis == 0:
        if delimiter is None:
            delimiter = arr.shape[0] // 2
        x_mask = arr[:delimiter, :]
        y_mask = arr[delimiter:, :]
        x_mask[-1, :] = 1
        y_mask[0, :] = 1
        return x_mask, y_mask
    else:
        if delimiter is None:
            delimiter = arr.shape[1] // 2
        x_mask = arr[:, :delimiter]
        y_mask = arr[:, delimiter:]
        x_mask[:, -1] = 1
        y_mask[:, 0] = 1
        return x_mask, y_mask


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


def solve_chunk(domain: np.ndarray, requirements) -> list:
    """ Solves the small chunk and returns the list of possible states """
    rows, cols = domain.shape

    if rows == 1 and cols == 1:
        out_vars = []
        for possible_value in domain[0, 0]:
            out_vars.append(np.reshape(possible_value, (1, 1)))
        return out_vars

    if rows > cols:
        axis = 0
    else:
        axis = 1

    x_domain, y_domain = split(domain, axis=axis)
    x_vars = solve_chunk(x_domain, requirements)
    y_vars = solve_chunk(y_domain, requirements)
    out_vars = []
    for x in x_vars:
        for y in y_vars:
            if check_if_can_be_merged(x, y, requirements, axis=axis):
                out_vars.append(merge(x, y, axis=axis))
    return out_vars


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

        for i, x in enumerate(solve_chunk(x_domain, requirements)):
            if i == 0:
                for y in solve_chunk(y_domain, requirements):
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


def get_index(i, j, cols, offset=0):
    return i * cols + j + offset


def find_constraints(shapes):
    constraints = []

    c_tuple = Constraints(l_constrained=set(),
                          r_constrained=set(),
                          u_constrained=set(),
                          d_constrained=set(),
                          lr_pairs=set(),
                          ud_pairs=set())

    offset = 0
    for shape_idx, shape in enumerate(shapes):
        shape_rows, shape_cols = shape
        print('\n{0}: rows={1} cols={2}'.format(shape_idx, shape_rows, shape_cols))
        for i in range(shape_rows):
            print('\t'.join([str(get_index(i, j, cols=shape_cols, offset=offset)) for j in range(shape_cols)]))
            for j in range(shape_cols):

                left = get_index(i, j - 1, cols=shape_cols, offset=offset)
                right = get_index(i, j, cols=shape_cols, offset=offset)
                up = get_index(i - 1, j, cols=shape_cols, offset=offset)
                down = get_index(i, j, cols=shape_cols, offset=offset)

                if j > 0:
                    c_tuple.lr_pairs.add((left, right))
                    c_tuple.l_constrained.add(right)  # has requirement of the left
                    c_tuple.r_constrained.add(left)  # has requirement of the right

                    constraints.append((right, left, LEFT))
                    constraints.append((left, right, RIGHT))

                if i > 0:
                    c_tuple.ud_pairs.add((up, down))
                    c_tuple.u_constrained.add(down)  # has requirement of the top
                    c_tuple.d_constrained.add(up)  # has requirement of the bottom

                    constraints.append((down, up, UP))
                    constraints.append((up, down, DOWN))

        offset += shape_rows * shape_cols

    constraints_dict = defaultdict(list)
    for src, tgt, direction in constraints:
        constraints_dict[src].append((tgt, direction))

    """
    for k, v in constraints_dict.items():
        print(k, v)
    """

    print('Left-Right pairs')
    print(c_tuple.lr_pairs)

    print('Up-Down pairs')
    print(c_tuple.ud_pairs)

    return constraints_dict, c_tuple


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


def reduce_domain_ac3(domain: np.ndarray, constraints):
    rows, cols = domain.shape
    q = []

    it = np.nditer(domain, flags=['multi_index', 'refs_ok'])
    while not it.finished:
        i, j = it.multi_index

        consistent = True
        if i > 0 and remove_inconsistent(i, j, i - 1, j, domain, constraints):
            consistent = False
        if i < rows - 1 and remove_inconsistent(i, j, i + 1, j, domain, constraints):
            consistent = False
        if j > 0 and remove_inconsistent(i, j, i, j - 1, domain, constraints):
            consistent = False
        if j < cols - 1 and remove_inconsistent(i, j, i, j + 1, domain, constraints):
            consistent = False

        if not consistent:
            q += neighbors(i, j, rows, cols)

        it.iternext()

    while len(q) > 0:
        i_src, j_src, i_tgt, j_tgt = q.pop(0)
        if remove_inconsistent(i_src, j_src, i_tgt, j_tgt, domain, constraints):
            q += neighbors(i_src, j_src, rows, cols)


def remove_inconsistent(i_src, j_src, i_tgt, j_tgt, domain: np.ndarray, constraints):
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
            print('\t[AC3] removed val {0} from ({1}, {2})'.format(v_src, i_src, j_src))
            domain[i_src, j_src].remove(v_src)
            removed = True

    return removed


def breakdown(r1, r2, c1, c2, depth: int, tasks: list):
    cols = c2 - c1
    rows = r2 - r1
    axis = 0
    delimiter = None
    subtask_a = None
    subtask_b = None

    task = {
        'id': -1,
        'rect': (r1, r2, c1, c2),
        'depth': depth,
        'merge_a': None,
        'merge_b': None,
        'merge_axis': None
    }

    if rows * cols < 16:
        task['id'] = len(tasks)
        tasks.append(task)
        return task

    if rows > cols:
        axis = 0
        delimiter = rows // 2
        subtask_a = r1, r1 + delimiter, c1, c2
        subtask_b = r1 + delimiter, r2, c1, c2
    else:
        axis = 1
        delimiter = cols // 2
        subtask_a = r1, r2, c1, c1 + delimiter
        subtask_b = r1, r2, c1 + delimiter, c2

    ta = breakdown(subtask_a[0], subtask_a[1], subtask_a[2], subtask_a[3], depth + 1, tasks)
    tb = breakdown(subtask_b[0], subtask_b[1], subtask_b[2], subtask_b[3], depth + 1, tasks)

    task['id'] = len(tasks)
    task['merge_a'] = ta['id']
    task['merge_b'] = tb['id']
    task['merge_axis'] = axis

    tasks.append(task)
    return task


def schedule(shape: tuple):
    tasks = []
    breakdown(0, shape[0], 0, shape[1], 0, tasks)
    tasks_by_depth = {}

    for t in tasks:
        depth = t['depth']
        if depth not in tasks_by_depth:
            tasks_by_depth[depth] = {}

        tasks_by_depth[depth][t['id']] = t

    return tasks_by_depth


def merge_task(x_vars: list, y_vars: list, axis, requirements: Constraints):
    out_vars = []
    for x in x_vars:
        for y in y_vars:
            if check_if_can_be_merged(x, y, requirements, axis=axis):
                out_vars.append(merge(x, y, axis=axis))
    print('\t\tMerged {0} values with {1}, resulting with {2} variants'.format(len(x_vars), len(y_vars), len(out_vars)))
    return out_vars


def solve_task(domain, requirements: Constraints):
    out_vars = list(iterate_solutions(domain, requirements))
    print('\t\tSolved {0} resulting {1} variants'.format(domain.shape, len(out_vars)))
    return out_vars


def run_tasks(tasks, domain, requirements: Constraints):
    depth_levels = sorted(list(tasks.keys()))
    storage = {}

    for depth_level in reversed(depth_levels):
        pool = multiprocessing.Pool()
        async_results = []
        print('Solving level: {0}'.format(depth_level))

        current_level_tasks = tasks[depth_level].values()
        print('\tTasks: {0}'.format(len(current_level_tasks)))

        for task in current_level_tasks:
            if task['merge_axis'] is None:
                """ Solve task """
                r1, r2, c1, c2 = task['rect']
                task_result = pool.apply_async(solve_task, kwds={
                    'domain': domain[r1:r2, c1:c2],
                    'requirements': requirements
                })
                async_results.append((task['id'], task_result))
            else:
                """ Merge task """
                task_result = pool.apply_async(merge_task, kwds={
                    'x_vars': storage[task['merge_a']],
                    'y_vars': storage[task['merge_b']],
                    'axis': task['merge_axis'],
                    'requirements': requirements
                })
                async_results.append((task['id'], task_result))

        pool.close()
        pool.join()

        print('\tDone')
        print('\tStoring results...')
        for task_id, task_result in async_results:
            storage[task_id] = task_result.get()
        print('\tDone')

    return storage


def solve(pizza, ingredients, max_size):
    tasks = schedule(pizza.shape)

    print('Getting domain of possible values...')
    domain = dm.get_domain(pizza, ingredients, max_size)

    print('Computing constraints...')
    constraints, c_tuple = find_constraints(domain.shapes)

    print('Reducing inconsistent domain...')
    reduce_domain_ac3(domain.domain, c_tuple)

    print('Solving...')
    #min_conflicts_solve(domain.domain, c_tuple)
    wave_function_collapse(domain.domain, c_tuple)
    #results = run_tasks(tasks=tasks, domain=domain.domain, requirements=c_tuple)
    #print(results)

    """
    solutions = iterate_solutions(domain.domain.view()[:3, :3], c_tuple)
    n = 0
    for s in solutions:
        print(s)
        n += 1
    print(n)
    """


def min_conflicts_solve(domain: np.ndarray, requirements: Constraints):
    rows, cols = domain.shape
    state = greedy_guess(domain, requirements)
    conflict_map = np.zeros(domain.shape, dtype=SOLUTION_DTYPE)

    print('Initial state...')
    for i in range(rows):
        for j in range(cols):
            state[i, j] = np.random.choice(domain[i, j])

    wnd = viz.Window(1200, 650)
    state_drawer = wnd.add_drawer(viz.ImgPlotDrawer(viz.Rect(0, 0, 512, 512)))
    conflict_drawer = wnd.add_drawer(viz.ImgPlotDrawer(viz.Rect(550, 0, 512, 512)))
    state_drawer.set_value(state)
    conflict_drawer.set_value(conflict_map)

    while True:
        conflicting = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                left = state[i, j]
                up = state[i, j]
                right = state[i, j + 1]
                down = state[i + 1, j]

                if not is_valid_assignment_top_bottom(up, down, requirements):
                    conflicting.append((i, j))
                    conflict_map[i, j] = 1
                elif not is_valid_assignment_left_right(left, right, requirements):
                    conflicting.append((i, j))
                    conflict_map[i, j] = 1
                else:
                    conflict_map[i, j] = 0



        if len(conflicting) == 0:
            print(state)
            return state

        # Select random conflicting variable
        conflicting_i, conflicting_j = conflicting[np.random.choice(len(conflicting))]

        # Count conflicts for each possible assignment
        min_conf = 9999999999999
        min_conf_val = None

        for val in domain[conflicting_i, conflicting_j]:
            conflicts = 0

            if conflicting_j < cols - 1:
                if not is_valid_assignment_left_right(val, state[conflicting_i, conflicting_j + 1], requirements):
                    conflicts += 1
            if conflicting_j > 0:
                if not is_valid_assignment_left_right(state[conflicting_i, conflicting_j - 1], val, requirements):
                    conflicts += 1
            if conflicting_i < rows - 1:
                if not is_valid_assignment_top_bottom(val, state[conflicting_i + 1, conflicting_j], requirements):
                    conflicts += 1
            if conflicting_i > 0:
                if not is_valid_assignment_top_bottom(state[conflicting_i - 1, conflicting_j], val, requirements):
                    conflicts += 1

            if conflicts <= min_conf:
                min_conf = conflicts
                min_conf_val = val

        state[conflicting_i, conflicting_j] = min_conf_val
        state_drawer.set_value(state)
        conflict_drawer.set_value(conflict_map)
        wnd.draw()


def greedy_guess(domain_raw: np.ndarray, requirements: Constraints):
    import pickle
    rows, cols = domain_raw.shape
    state = np.empty(domain_raw.shape, dtype=SOLUTION_DTYPE)
    domain = pickle.loads(pickle.dumps(domain_raw))

    for i in range(rows):
        for j in range(cols):
            choices = domain[i, j]
            if len(choices) == 0:
                continue
            val = np.random.choice(choices)
            state[i, j] = val

            if i < rows - 1:
                for down in domain[i+1, j]:
                    if not is_valid_assignment_top_bottom(val, down, requirements):
                        domain[i+1, j].remove(down)

            if j < cols - 1:
                for right in domain[i, j+1]:
                    if not is_valid_assignment_left_right(val, right, requirements):
                        domain[i, j+1].remove(right)
    return state


def wave_function_collapse(domain_raw: np.ndarray, requirements: Constraints):
    import pickle
    import time
    rows, cols = domain_raw.shape
    state = np.zeros(domain_raw.shape, dtype=SOLUTION_DTYPE)
    entropy = np.zeros(domain_raw.shape, dtype=np.int16)
    domain = pickle.loads(pickle.dumps(domain_raw))

    for i in range(rows):
        for j in range(cols):
            entropy[i, j] = len(domain[i, j])

    wnd = viz.Window(1200, 650)
    state_drawer = wnd.add_drawer(viz.ImgPlotDrawer(viz.Rect(0, 0, 512, 512)))
    entropy_drawer = wnd.add_drawer(viz.ImgPlotDrawer(viz.Rect(550, 0, 512, 512)))
    state_drawer.set_value(state)
    entropy_drawer.set_value(entropy)

    while True:
        nonzero_entropy = entropy[np.nonzero(entropy)]
        if len(nonzero_entropy) == 0:
            return state

        i, j = np.unravel_index(np.argmin(nonzero_entropy), entropy.shape)
        valid_values = domain[i, j]
        if len(valid_values) == 0:
            raise RuntimeError('WTF, invalid entropy')

        chosen = valid_values[np.random.randint(len(valid_values))]
        state[i, j] = chosen

        entropy[i, j] = 0
        domain[i, j] = []

        if i > 0:
            for up in domain[i - 1, j]:
                if not is_valid_assignment_top_bottom(up, chosen, requirements):
                    domain[i - 1, j].remove(up)
                    entropy[i - 1, j] -= 1

        if i < rows - 1:
            for down in domain[i + 1, j]:
                if not is_valid_assignment_top_bottom(chosen, down, requirements):
                    domain[i + 1, j].remove(down)
                    entropy[i + 1, j] -= 1

        if j > 0:
            for left in domain[i, j - 1]:
                if not is_valid_assignment_left_right(left, chosen, requirements):
                    domain[i, j - 1].remove(left)
                    entropy[i, j - 1] -= 1

        if j < cols - 1:
            for right in domain[i, j + 1]:
                if not is_valid_assignment_left_right(chosen, right, requirements):
                    domain[i, j + 1].remove(right)
                    entropy[i, j + 1] -= 1

        reduce_domain_ac3(domain, requirements)
        for i in range(rows):
            for j in range(cols):
                entropy[i, j] = len(domain[i, j])

        entropy_drawer.set_value(entropy)
        state_drawer.set_value(state)
        wnd.draw()
