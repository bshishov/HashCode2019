import pickle
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
import utils
import os


def __create_domain(rows, cols):
    state = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            state[i, j] = []
    return state


def analyze_shape(pizza: np.ndarray,
                  min_ingredients: int,
                  max_slice: int,
                  offset: int,
                  shape_cols: int,
                  shape_rows: int):
    rows, cols = pizza.shape
    entropy = np.zeros(pizza.shape, dtype=np.uint16)
    domain = __create_domain(rows, cols)

    for i in range(rows):
        for j in range(cols):
            r1 = i
            r2 = i + shape_rows
            c1 = j
            c2 = j + shape_cols
            if utils.is_valid_slice(pizza, r1, r2, c1, c2, min_ingredients, max_slice):
                entropy[r1:r2, c1:c2] += 1

                # Mark all the cells inside slice that they could be potentially sliced by this slice
                for i2 in range(r1, r2):
                    for j2 in range(c1, c2):
                        i_local = i2 - r1
                        j_local = j2 - c1
                        domain[i2, j2].append(offset + i_local * shape_cols + j_local)

    # Return all values
    return entropy, domain, offset


def analyze(pizza: np.ndarray,
            min_ingredients: int,
            max_slice: int,
            pool_class=multiprocessing.Pool):
    valid_shapes = utils.get_possible_shapes(min_ingredients, max_slice)
    for shape_idx, shape in enumerate(valid_shapes):
        print('Shape {0}: {1}'.format(shape_idx, shape))

    rows, cols = pizza.shape
    domain = __create_domain(rows, cols)

    entropy = np.zeros(pizza.shape, dtype=np.uint16)

    pool = pool_class()
    results = []
    offset = 0
    for shape_rows, shape_cols in valid_shapes:
        r = pool.apply_async(analyze_shape, kwds={
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
        local_entropy, local_state, local_shape_idx = r.get()
        entropy += local_entropy

        # State aggregation
        for i in range(rows):
            for j in range(cols):
                assert local_entropy[i, j] == len(local_state[i, j])
                domain[i, j] += local_state[i, j]
    return valid_shapes, entropy, domain


def load(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


class Domain(object):
    def __init__(self):
        self.domain = None
        self.shapes = None
        self.entropy = None

    def calculate(self, pizza: np.ndarray, min_ingredients: int, max_slice_size: int):
        self.shapes, self.entropy, self.domain = analyze(pizza, min_ingredients, max_slice_size)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


def get_domain(pizza, ingredients, max_size):
    domain_file_name = '{0}x{1}.domain'.format(pizza.shape[0], pizza.shape[1])

    if os.path.exists(domain_file_name):
        print('Loading domain: {0}'.format(domain_file_name))
        return load(domain_file_name)

    print('Analyzing domain')
    d = Domain()
    d.calculate(pizza, min_ingredients=ingredients, max_slice_size=max_size)
    print('Saving to: {0}'.format(domain_file_name))
    d.save(domain_file_name)
    return d
