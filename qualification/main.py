import argparse
import numpy as np


def solve(*args, **kwargs):
    return None


def main(args):
    print('Solving')
    print('Input: {0}'.format(args.input))
    print('Output: {0}'.format(args.output))

    with open(args.input, 'r') as in_file:import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils.parallel as parallel


def preprocess(idx, photo1: tuple, all_photos: list):
    n = len(all_photos)
    intersections = np.zeros(n, np.int16)
    diffsizehalf = np.zeros(n, np.int16)
    u = np.zeros(n, np.int16)

    tags1 = photo1[3]
    for i, photo2 in enumerate(all_photos):
        if idx == i:
            continue
        tags2 = photo2[3]
        num_intersections = len(tags1.intersection(tags2))
        if num_intersections == 0:
            continue

        h = int(np.abs(len(tags1) - len(tags2))) // 2
        intersections[i] = num_intersections
        diffsizehalf[i] = h
        if photo2[1]:
            u[i] = intersections[i]
        else:
            u[i] = np.maximum(intersections[i] - diffsizehalf[i], 0)
    return idx, intersections, diffsizehalf, u


def merge(v1, v2):
    assert v1[1] == v2[1] == True
    merged_tags = v1[3].union(v2[3])
    return [v1[0], v2[0]], False, len(merged_tags), merged_tags

#def select_vertical(photos):




def solve(photos):
    tags_set = set()
    n = len(photos)
    intersections = np.zeros((n, n), dtype=np.int16)
    diffsizehalf = np.zeros((n, n), dtype=np.int16)

    """
    for id, is_vertical, num_tags, tags in photos:
        tags_set.intersection_update(tags)
        for t in tags:
            tags_set.add(t)
    """

    sorted_by_num_tags = sorted(photos, key=lambda v: v[2])

    """
    p = parallel.ParallelExecutor('thread')
    with p:
        for i in range(n - 1):
            p.dispatch(preprocess, i, sorted_by_num_tags[i], sorted_by_num_tags)

    for res in p.iterate_results():
        idx, intersections_local, diffsizehalf_local = res
        intersections[idx, :] = intersections_local
        
        diffsizehalf[idx, :] = diffsizehalf_local
    """

    results = []

    start_idx = 0
    #horizontal = list(filter(lambda p: p[1] == False, sorted_by_num_tags))
    #vertical = filter(lambda p: p[1] == True, sorted_by_num_tags)

    merged = []
    vertical = list(filter(lambda p: p[1], sorted_by_num_tags))
    while True:

        if len(vertical) == 1:
            vertical.remove(vertical[0])

        if len(vertical) == 0:
            break

        v1 = vertical[np.random.choice(len(vertical))]
        _, v1_intersections, _, _ = preprocess(v1[0], v1, vertical)

        non_zero_indices = np.nonzero(v1_intersections)[0]
        if len(non_zero_indices) > 0:
            tmp_idx = np.argmin(v1_intersections[non_zero_indices])
            v2_local_index = non_zero_indices[tmp_idx]

            v2 = vertical[v2_local_index]
            vertical.remove(v1)
            vertical.remove(v2)
            merged.insert(0, merge(v1, v2))
        else:
            vertical.remove(v1)

    sorted_by_num_tags = list(filter(lambda p: p[1] , sorted_by_num_tags)) + merged
    #plt.imshow(intersections)
    #plt.show()
    #print('Mean intersections: {0}'.format(np.mean(intersections)))
    #u = np.maximum(intersections - diffsizehalf, 0)

    include_mask = np.empty(len(sorted_by_num_tags), dtype=np.bool)
    indices = np.arange(len(sorted_by_num_tags))
    include_mask[:] = True
    current_global_idx = start_idx
    include_mask[0] = False
    results.append([sorted_by_num_tags[current_global_idx][0]])

    for iteration in range(len(sorted_by_num_tags) - 1):
        print(iteration)

        _, local_i, _, local_u = preprocess(current_global_idx, sorted_by_num_tags[current_global_idx], sorted_by_num_tags)

        local_i = local_i[include_mask]
        local_u = local_u[include_mask]
        local_indices = indices[include_mask]

        if np.max(local_i) == 0:
            break

        max_local_idx = np.argmax(local_u)
        if local_u[max_local_idx] <= 0:
            max_local_idx = np.argmax(local_i)

        if local_i[max_local_idx] == 0:
            break

        include_mask[current_global_idx] = False
        include_mask[local_indices[max_local_idx]] = False

        results.append(sorted_by_num_tags[local_indices[max_local_idx]][0])
        current_global_idx = local_indices[max_local_idx]
        print(current_global_idx)

    return results


def main(args):
    print('Solving')
    print('Input: {0}'.format(args.input))
    print('Output: {0}'.format(args.output))

    with open(args.input, 'r') as in_file:
        total_photos = int(in_file.readline())
        photos = []

        for i in range(total_photos):
            photo_args = in_file.readline().split()
            is_vertical = photo_args[0] == 'V'
            num_tags = int(photo_args[1])
            tags = set(photo_args[2:])
            photos.append(([i], is_vertical, num_tags, tags))

    slides = solve(photos)

    with open(args.output, 'w') as out_file:
        out_file.write(str(len(slides)))
        out_file.write('\n')
        for slide in slides:
            out_file.write(' '.join(map(str, slide)))
            out_file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input filename', required=True)
    parser.add_argument('--output', type=str, help='Output filename', required=True)
    main(args=parser.parse_args())
