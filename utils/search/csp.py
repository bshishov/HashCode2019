import numpy as np

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3


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


def reduce_domain_ac3(domain: np.ndarray, constraints: dict):
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


def remove_inconsistent(i_src, j_src, i_tgt, j_tgt, domain: np.ndarray, constraints: dict):
    removed = False
    for v_src in domain[i_src, j_src]:
        can_satisfy_constraint = False
        src_c = constraints[v_src]
        for v_tgt in domain[i_tgt, j_tgt]:
            x_dir = j_tgt - j_src
            y_dir = i_tgt - i_src

            for v_tgt_c, direction in src_c:
                if v_tgt == v_tgt_c:
                    if x_dir == 1 and direction == RIGHT:
                        can_satisfy_constraint = True
                        break
                    if x_dir == -1 and direction == LEFT:
                        can_satisfy_constraint = True
                        break
                    if y_dir == 1 and direction == DOWN:
                        can_satisfy_constraint = True
                        break
                    if y_dir == -1 and direction == UP:
                        can_satisfy_constraint = True
                        break

            if can_satisfy_constraint:
                break

        if not can_satisfy_constraint:
            print('AC3 REMOVED {0} from ({1}, {2})'.format(v_src, i_src, j_src))
            domain[i_src, j_src].remove(v_src)
            removed = True

    return removed


class CspError(RuntimeError):
    pass


class CspProblem(object):
    def iterate_variables(self):
        raise NotImplementedError

    def domain_values(self, var) -> list:
        raise NotImplementedError

    def remove_from_domain(self, var, val):
        raise NotImplementedError

    def get_constraints(self):
        raise NotImplementedError

    def iterate_neighbours(self, var) -> list:
        raise NotImplementedError

    def get_neighbours(self, var) -> list:
        return list(self.iterate_neighbours(var))

    def count_conflicts(self, var, val) -> int:
        raise NotImplementedError

    def values_conflicting(self, var1, val1, var2, val2) -> bool:
        raise NotImplementedError

    def var_conflicting(self, var) -> bool:
        for var2 in self.iterate_neighbours(var):
            if self.vars_conflicting(var, var2):
                return True
        return False

    def vars_conflicting(self, var1, var2) -> bool:
        return self.values_conflicting(var1, self.get_value(var1), var2, self.get_value(var2))

    def get_value(self, var):
        raise NotImplementedError

    def set_value(self, var, value):
        raise NotImplementedError

    def __getitem__(self, var):
        return self.get_value(var)

    def __setitem__(self, var, value):
        self.set_value(var, value)


def min_conflicts_step(csp: CspProblem):
    conflicting_variables = []
    for var in csp.iterate_variables():
        if csp.var_conflicting(var):
            conflicting_variables.append(var)

    if len(conflicting_variables) == 0:
        return

    var = conflicting_variables[np.random.randint(len(conflicting_variables))]

    vals = csp.domain_values(var)
    confs = [csp.count_conflicts(var, v) for v in vals]
    p = np.exp(1.0 - np.array(confs) / np.max(confs))
    idx = np.random.choice(len(vals), p=p / p.sum())
    csp.set_value(var, vals[idx])
    return confs[idx]

    min_conf = 9999999999999
    min_conf_val = None

    for val in csp.domain_values(var):
        conflicts = csp.count_conflicts(var, val)
        if conflicts <= min_conf:
            min_conf = conflicts
            min_conf_val = val

    csp.set_value(var, min_conf_val)


def arc_consistency3(csp: CspProblem):
    queue = []

    for x in csp.iterate_variables():
        for y in csp.iterate_neighbours(x):
            if arc_reduce(csp, x, y):
                if not csp.domain_values(x):
                    raise CspError('Empty domain after reduction')
                else:
                    for z in csp.iterate_neighbours(x):
                        if z != y:
                            queue.append((z, x))

    while len(queue) > 0:
        x, y = queue.pop(0)
        if arc_reduce(csp, x, y):
            if not csp.domain_values(x):
                raise CspError('Empty domain after reduction')
            else:
                for z in csp.iterate_neighbours(x):
                    if z != y:
                        queue.append((z, x))


def arc_reduce(csp: CspProblem, x, y):
    reduced = False
    for vx in csp.domain_values(x):
        can_satisfy_constraint = False
        for vy in csp.domain_values(y):
            if not csp.values_conflicting(x, vx, y, vy):
                can_satisfy_constraint = True
                break
        if not can_satisfy_constraint:
            csp.remove_from_domain(x, vx)
            reduced = True
    return reduced
