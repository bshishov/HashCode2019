import argparse
import numpy as np
import collections
import time
import random

vertical_photos = []
tagged_photos = {}
tags = collections.Counter()
slides = []

def solve(*args, **kwargs):
    pass


def group_vertical_photos():
    slides = []
    while len(vertical_photos) > 0:
        first = vertical_photos.pop()
        second = vertical_photos.pop()
        slide = {
            'id': first['id'] + second['id'],
            'tags': set(first['tags'] + second['tags']),
        }
        slides.append(slide)
    return slides

def main(args):
    global slides
    global vertical_photos

    print('Solving')
    print('Input: {0}'.format(args.input))
    print('Output: {0}'.format(args.output))

    with open(args.input, 'r') as in_file:
        task_args = list(map(int, in_file.readline().split()))

        for i in range(0, task_args[0]):
            photo_data = in_file.readline().split()

            photo = {
                'id': (i,),
                # 'tags_size': int(photo_data[1]),
                'tags': photo_data[2:]
            }

            if photo_data[0] == 'V':
                vertical_photos.append(photo)
            else:
                photo['tags'] = set(photo['tags'])
                slides.append(photo)
    print('have read')

    slides = slides + group_vertical_photos()
    print('have grouped')

    for slide in slides:
        for tag in slide['tags']:
            tags[tag] += 1

    sorted_tags = tags.most_common()
    # print(sorted_tags)
    print('sorted tags')

    arr_len = len(slides)
    intersections = []
    for i in range(0, arr_len):
        start_time = time.time()
        item = []
        for j in range(i+1, arr_len):
            # print(slides[i]['tags'])
            # print(slides[j]['tags'])
            is_intersected = len(slides[i]['tags'] & slides[j]['tags'])
            # is_intersected = random.randint(0, 100)

            if is_intersected > 0:
                item.append(j)
                if len(item) > 1:
                    break

        elapsed_time = time.time() - start_time
        print('round: ' + str(i) + ' ' + str(elapsed_time))
        intersections.append(item)
    print('intersections')

    # по строкам
    #intersections_sums = intersections.sum(1)
    print('sums')
    #intersections_maxs = intersections.max(1)
    print('maxs')

    checked_slides = np.ones([arr_len])
    solution = []
    print('checked slides')
    while checked_slides.sum() != 0:
        all_non_zeroes = np.nonzero(checked_slides)[0]

        if len(all_non_zeroes) == 0:
            break

        # находим первый не нулевой
        first = all_non_zeroes[0]
        checked_slides[first] = 0
        solution.append(first)

        # if intersections_sums[first] == 0:
        #    continue

        #
        start_time = time.time()
        while True:
            # находим первое пересмечени (большое)
            print(first)

            found = False
            for z in intersections[first]:
                if checked_slides[z] != 0:
                    checked_slides[z] = 0
                    solution.append(z)
                    first = z
                    found = True
                    break
                else:
                    continue
            if not found:
                break
        elapsed_time = time.time() - start_time
        print('round: ' + str(elapsed_time))
    # print(solution)

    for slide in slides:
        print(slide)

    with open(args.output, 'w') as out_file:
        out_file.write(str(len(solution)) + '\n')
        for s in solution:
            string = ' '.join((str(i) for i in slides[s]['id']))
            out_file.write(string + '\n')
        # TODO: output solution
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input filename', required=True)
    parser.add_argument('--output', type=str, help='Output filename', required=True)
    main(args=parser.parse_args())
