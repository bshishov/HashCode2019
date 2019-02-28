from __future__ import annotations
import math

from queue import LifoQueue, Queue
from typing import Iterable, Tuple, List, Any, Optional, Callable

# Basic types
Action = Any
State = Any
GoalTestFn = Callable[[State], bool]
SuccessorFn = Callable[[State], Iterable[Tuple[Action, State, float]]]
HeuristicFn = Callable[[State], float]


class SortedQueue(object):
    # TODO: improve performance with heapq
    def __init__(self):
        self._items = []

    def qsize(self):
        return len(self._items)

    def get(self):
        return self._items.pop(0)

    def put(self, item):
        for i, el in enumerate(self._items):
            if item < el:
                self._items.insert(i, item)
                return

        self._items.append(item)

    def remove(self, item):
        self._items.remove(item)


class SearchFailure(RuntimeError):
    pass


class NoSolutionFound(SearchFailure):
    pass


class TreeNode(object):
    def __init__(self, state: State,
                 parent: TreeNode=None,
                 action: Action=None,
                 step_cost: float=0,
                 heuristic: Optional[float]=None):

        if parent is not None:
            self.depth = parent.depth + 1
            self.path_cost = parent.path_cost + step_cost
        else:
            self.depth = 0
            self.path_cost = step_cost

        self.heuristic = heuristic
        self.state = state
        self.parent = parent
        self.action = action
        if heuristic is not None:
            self.f_cost = self.path_cost + heuristic
        else:
            self.f_cost = self.path_cost

    def unroll(self) -> List[Tuple[Action, State]]:
        history = []
        node = self
        while node is not None:
            if node.action is not None:
                history.append((node.action, node.state))
            node = node.parent
        return list(reversed(history))

    def __gt__(self, other: TreeNode):
        return self.f_cost > other.f_cost

    def __lt__(self, other: TreeNode):
        return self.f_cost < other.f_cost


def iterate_successors(node: TreeNode,
                       successor_fn: SuccessorFn) -> Iterable[TreeNode]:
    for state, action, cost in successor_fn(node.state):
        yield TreeNode(state, parent=node, action=action, step_cost=cost)


def iterate_successors_with_heuristic(node: TreeNode,
                                      successor_fn: SuccessorFn,
                                      heuristic_fn: HeuristicFn):
    for state, action, cost in successor_fn(node.state):
        yield TreeNode(state,
                       parent=node,
                       action=action,
                       heuristic=heuristic_fn(state),
                       step_cost=cost)


def tree_search(initial_state: State,
                successor_fn: SuccessorFn,
                goal_test_fn: GoalTestFn,
                queue_cls: type) -> TreeNode:
    frontier = queue_cls()
    frontier.put(TreeNode(initial_state))
    while True:
        if frontier.qsize() == 0:
            raise NoSolutionFound()
        node = frontier.get()
        if goal_test_fn(node.state):
            return node
        for successor in iterate_successors(node, successor_fn):
            frontier.put(successor)


def depth_tree_search(initial_state: State,
                      successor_fn: SuccessorFn,
                      goal_test_fn: GoalTestFn) -> TreeNode:
    return tree_search(initial_state=initial_state,
                       successor_fn=successor_fn,
                       goal_test_fn=goal_test_fn,
                       queue_cls=LifoQueue)


def broad_tree_search(initial_state: State,
                      successor_fn: SuccessorFn,
                      goal_test_fn: GoalTestFn) -> TreeNode:
    return tree_search(initial_state=initial_state,
                       successor_fn=successor_fn,
                       goal_test_fn=goal_test_fn,
                       queue_cls=Queue)


def graph_search(initial_state: State,
                 successor_fn: SuccessorFn,
                 goal_test_fn: GoalTestFn,
                 queue_cls: type) -> TreeNode:
    closed_set = set()
    open_set = queue_cls()
    open_set.put(TreeNode(initial_state))
    while True:
        if open_set.qsize() == 0:
            raise NoSolutionFound()
        node = open_set.get()
        if goal_test_fn(node.state):
            return node

        if node.state in closed_set:
            continue

        closed_set.add(node.state)

        for successor in iterate_successors(node, successor_fn):
            open_set.put(successor)


def depth_graph_search(initial_state: State,
                       successor_fn: SuccessorFn,
                       goal_test_fn: GoalTestFn) -> TreeNode:
    return graph_search(initial_state=initial_state,
                        successor_fn=successor_fn,
                        goal_test_fn=goal_test_fn,
                        queue_cls=LifoQueue)


def broad_graph_search(initial_state: State,
                       successor_fn: SuccessorFn,
                       goal_test_fn: GoalTestFn) -> TreeNode:
    return graph_search(initial_state=initial_state,
                        successor_fn=successor_fn,
                        goal_test_fn=goal_test_fn,
                        queue_cls=Queue)


def uniform_cost_search(initial_state: State,
                        successor_fn: SuccessorFn,
                        goal_test_fn: GoalTestFn) -> TreeNode:
    closed_set_states = set()
    open_set_nodes = SortedQueue()
    open_set_states = {}

    node = TreeNode(initial_state)
    open_set_nodes.put(node)
    open_set_states[node.state] = node

    while True:
        if open_set_nodes.qsize() == 0:
            raise NoSolutionFound

        # Get node from open queue
        node = open_set_nodes.get()

        # Check solution
        if goal_test_fn(node.state):
            return node

        # Mark state as already visited
        closed_set_states.add(node.state)

        for successor in iterate_successors(node, successor_fn):
            # check whether node with this state already exists
            existing = open_set_states.get(successor.state, None)

            if existing is None and successor.state not in closed_set_states:
                # if node is explored and does not exist
                open_set_nodes.put(successor)
                open_set_states[successor.state] = successor
            elif existing is not None and successor < existing:
                # if node exists in frontier and its more expensive that current node
                # than replace it with the new better one
                open_set_nodes.remove(existing)
                open_set_nodes.put(successor)


def a_star_search(initial_state: State,
                  successor_fn: SuccessorFn,
                  goal_test_fn: GoalTestFn,
                  heuristic_fn: HeuristicFn) -> TreeNode:
    closed_set_states = set()
    open_set_nodes = SortedQueue()
    open_set_states = {}

    node = TreeNode(initial_state, heuristic=heuristic_fn(initial_state))
    open_set_nodes.put(node)
    open_set_states[node.state] = node

    while open_set_nodes.qsize() > 0:
        # Get node from open queue
        node = open_set_nodes.get()

        # Check solution
        if goal_test_fn(node.state):
            return node

        # Mark state as already visited
        closed_set_states.add(node.state)

        for successor in iterate_successors_with_heuristic(node, successor_fn, heuristic_fn):
            # Check whether node with this state already exists
            existing = open_set_states.get(successor.state, None)

            if existing is None and successor.state not in closed_set_states:
                # if node is explored and does not exist
                open_set_nodes.put(successor)
                open_set_states[successor.state] = successor
            elif existing is not None and successor < existing:
                # if node exists in frontier and its more expensive that current node
                # than replace it with the new better one
                open_set_nodes.remove(existing)
                open_set_nodes.put(successor)

    raise NoSolutionFound


def rbfs(node: TreeNode,
         goal_test_fn: GoalTestFn,
         heuristic_fn: HeuristicFn,
         successor_fn: SuccessorFn,
         f_limit: float) -> Tuple[Optional[TreeNode], float]:

    # If current node is a solution - return it up
    if goal_test_fn(node.state):
        return node, node.f_cost

    # Gather all successors of current node
    successors = []
    for state, action, step_cost in successor_fn(node.state):
        successor = TreeNode(state,
                             parent=node,
                             action=action,
                             step_cost=step_cost,
                             heuristic=heuristic_fn(state))
        successor.f_cost = max(successor.path_cost + successor.heuristic, node.f_cost)
        successors.append(successor)

    # If no successors then skip current branch
    if len(successors) == 0:
        return None, math.inf

    while True:
        # Sort successors by their f value
        open_list = sorted(successors, key=lambda s: s.f_cost)

        # Find the best successor
        best = open_list[0]

        # If the best is actually bad then skip this branch
        if best.f_cost > f_limit:
            return None, best.f_cost

        # Second alternative node
        f_alternative = math.inf
        if len(open_list) > 1:
            f_alternative = open_list[1].f_cost
        result, new_f = rbfs(node=best,
                             goal_test_fn=goal_test_fn,
                             heuristic_fn=heuristic_fn,
                             successor_fn=successor_fn,
                             f_limit=min(f_limit, f_alternative))
        best.f_cost = new_f

        if result is not None:
            return result, result.f_cost


def recursive_best_first_search(initial_state: State,
                                heuristic_fn: HeuristicFn,
                                goal_test_fn: GoalTestFn,
                                successor_fn: SuccessorFn):
    node, _ = rbfs(node=TreeNode(initial_state, heuristic=heuristic_fn(initial_state)),
                   goal_test_fn=goal_test_fn,
                   heuristic_fn=heuristic_fn,
                   successor_fn=successor_fn,
                   f_limit=math.inf)
    return node
