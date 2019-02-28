from practice.solver import Solver
import utils.viz as viz
import numpy as np
import time
import math


def solve(pizza, ingredients, max_size):
    solver = Solver(min_ingredients=ingredients, max_slice_size=max_size)
    solver.explain_shapes()
    solver.explain_constraints()

    wnd = viz.Window(600, 600)
    progress_drawer = wnd.add_drawer(viz.ImageDrawer(viz.Rect(100, 100, 400, 400)))
    progress = np.zeros(pizza.shape, dtype=np.uint8)

    rows, cols = pizza.shape
    chunk_width = 14
    chunk_height = 3

    steps_i = math.ceil(rows / chunk_height)
    steps_j = math.ceil(cols / chunk_width)

    for si in range(steps_i):
        for sj in range(steps_j):
            i = si * chunk_height
            j = sj * chunk_width

            part = pizza[i:i+chunk_height, j:j+chunk_width]
            solution = solver.first_solution(part, ac3_reduction=True, verbose=False)
            progress[i:i+chunk_height, j:j+chunk_width] = solution > 0
            progress_drawer.set_value(progress)
            wnd.draw()

    while True:
        wnd.draw()
        time.sleep(0.1)

    """
    for solution in solver.iterate_solutions(pizza[:6, :6], ac3_reduction=True, verbose=False):
        print('Solution')
        print(solution)
        print('Slices:')
        for slice in solver.iterate_slices(solution):
            print(slice)
    """