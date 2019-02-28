def test_queens(x1, y1, x2, y2):
    # vertical
    if x1 == x2:
        return False

    # horisontal
    if y1 == y2:
        return False

    # positive diag
    if y2 > y1 and x2 - x1 == y2 - y1:
        return False

    # negative diag
    if y2 < y1 and x2 - x1 == y1 - y2:
        return False

    return True


def under_attack_heuristic(state: tuple):
    under_attack = 0
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            if not test_queens(i, state[i], j, state[j]):
                under_attack += 1
    return under_attack


class NQueensProblem(object):
    def __init__(self, n: int = 5):
        self.n = n
        self.initial_state = ()

    def iterate_successors(self, state):
        for i in range(self.n):
            # Skip already placed queen-rows
            if i not in state:
                new_state = state + (i, )
                action = i
                cost = 0
                yield new_state, action, cost

    def goal_test(self, state):
        if len(state) != self.n:
            return False

        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                if not test_queens(i, state[i], j, state[j]):
                    return False
        return True


def main():
    import search.search as search

    problem = NQueensProblem(10)
    #solution = search.depth_tree_search(problem.initial_state, problem.iterate_successors, problem.goal_test)
    #solution = search.depth_graph_search(problem.initial_state, problem.iterate_successors, problem.goal_test)
    #solution = search.uniform_cost_search(problem.initial_state, problem.iterate_successors, problem.goal_test)

    solution = search.recursive_best_first_search(initial_state=problem.initial_state,
                                                  goal_test_fn=problem.goal_test,
                                                  successor_fn=problem.iterate_successors,
                                                  heuristic_fn=under_attack_heuristic)

    """
    solution = search.a_star_search(initial_state=problem.initial_state,
                                    goal_test_fn=problem.goal_test,
                                    successor_fn=problem.iterate_successors,
                                    heuristic_fn=under_attack_heuristic)
    """
    for i, (action, state) in enumerate(solution.unroll()):
        print('i={0}, action={1}, state={2}'.format(i, action, state))


if __name__ == '__main__':
    main()
