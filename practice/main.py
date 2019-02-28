import sys
import numpy as np

TOMATO = -1
MUSHROOM = 1


def solve(pizza: np.ndarray, ingredients: int, max_size: int) -> list:
    import practice.attempt5 as attempt5
    """ YOUR SOLUTION GOES HERE """
    return attempt5.solve(pizza, ingredients, max_size)


def main(*args):
    filename = args[0]
    print(f'Loading {filename}')

    with open(filename, 'r') as f:
        args = list(map(int, f.readline().split()))

        # Rectangular pizza dimensions, [R x C]
        rows = args[0]
        columns = args[1]

        # L - minimum number of each ingredient cells in a slice
        ingredients = args[2]

        # H - Maximum total number of cells of a slice
        slice_cap = args[3]

        # Allocate an array for pizza
        pizza = np.empty(shape=(rows, columns), dtype=np.int)

        # Read pizza
        def __to_ingredient(_val):
            if _val == 'T':
                return TOMATO
            return MUSHROOM

        for i in range(rows):
            row = list(map(__to_ingredient, f.readline()[:columns]))
            pizza[i, :] = row

    print(f'Pizza size: {pizza.shape}')
    print('Solving')
    solution = solve(pizza, ingredients, slice_cap)

    if solution:
        print('Total slices: {0}'.format(len(solution)))
        with open(filename + '.out', 'w') as out_file:
            out_file.write('{0}\n'.format(len(solution)))
            for r1, c1, r2, c2 in solution:
                print('Slice: ', r1, c1, r2, c2)
                out_file.write('{0} {1} {2} {3}\n'.format(r1, c1, r2, c2))
    else:
        print('No solution')


if __name__ == '__main__':
    main(*sys.argv[1:])
