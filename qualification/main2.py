import argparse
import numpy as np
import collections

number_of_photos = 0
photos = []
tags = collections.Counter()

def solve(*args, **kwargs):
    return None


def main(args):
    print('Solving')
    print('Input: {0}'.format(args.input))
    print('Output: {0}'.format(args.output))

    with open(args.input, 'r') as in_file:
        task_args = list(map(int, in_file.readline().split()))
        number_of_photos = task_args[0]

        for i in range(0, number_of_photos):
            photo = in_file.readline().split()

            photos.append({
                'orientation': photo[0],
                'tags_size': int(photo[1]),
                'tags': photo[2:]
            })

            for tag in photos[-1]['tags']:
                tags[tag] += 1

        sorted_tags = tags.most_common()

        print(sorted_tags)

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
