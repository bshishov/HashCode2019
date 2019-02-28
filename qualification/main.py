import argparse
import numpy as np


def solve(*args, **kwargs):
    return None


def main(args):
    print('Solving')
    print('Input: {0}'.format(args.input))
    print('Output: {0}'.format(args.output))

    with open(args.input, 'r') as in_file:
        task_args = list(map(int, in_file.readline().split()))
        # TODO: fill according to the task
        pass

    solution = solve(*task_args)

    with open(args.output, 'w') as out_file:
        # TODO: output solution
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input filename', required=True)
    parser.add_argument('--output', type=str, help='Output filename', required=True)
    main(args=parser.parse_args())
