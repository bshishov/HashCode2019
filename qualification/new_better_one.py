import argparse
import numpy as np
from typing import List, Iterable, Tuple
from dataclasses import dataclass
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm


@dataclass
class Photo:
    id: List[int]
    is_vertical: bool
    num_tags: int
    tags: set


def analyze(photo1: Photo, photo2: Photo) -> Tuple[int, int]:
    intersections = len(photo1.tags.intersection(photo2.tags))

    if intersections == 0:
        return 0, 0

    score = min(len(photo1.tags.difference(photo2.tags)),
                intersections,
                len(photo2.tags.difference(photo1.tags)))

    #h = int(np.abs(len(photo1.tags) - len(photo2.tags))) // 2
    #score = max(intersections - h, 0)
    return intersections, score


def process(photo: Photo, photos: List[Photo], pool: ThreadPool):
    n = len(photos)
    intersections = np.zeros(n, dtype=np.int8)
    scores = np.zeros(n, dtype=np.int8)

    """
    fn = partial(analyze, photo2=photo)
    results = pool.map(fn, photos)
    for i, res in enumerate(results):
        intersections[i], scores[i] = res
    """

    """
    for i, photo2 in enumerate(photos):
        intersections[i], scores[i] = analyze(photo, photo2)
    """

    for i, photo2 in enumerate(photos):
        ii = len(photo.tags.intersection(photo2.tags))

        if ii == 0:
            continue

        intersections[i] = ii
        scores[i] = min(len(photo.tags.difference(photo2.tags)),
                        ii,
                        len(photo2.tags.difference(photo.tags)))

    return intersections, scores


def get_random(arr: List) -> Tuple[int, object]:
    random_index = np.random.randint(len(arr))
    return random_index, arr[random_index]


def solve(photos: List[Photo]) -> List[Photo]:
    photos = sorted(photos, key=lambda x: x.num_tags)
    #current_photo = photos.pop(np.random.randint(len(photos)))
    current_photo = photos.pop(0)
    results = [current_photo]
    pool = ThreadPool()

    for _ in tqdm(range(len(photos))):
        if len(photos) == 0:
            break

        intersections, scores = process(current_photo, photos, pool)

        if np.max(scores) == 0:
            #current_photo = photos.pop(np.random.randint(len(photos)))
            current_photo = photos.pop(0)
            results.append(current_photo)
            continue

        best_score_idx = np.argmax(scores)
        photo2 = photos.pop(best_score_idx)
        results.append(photo2)
        current_photo = photo2

    return results


def main(args):
    print('Solving')
    print('Input: {0}'.format(args.input))
    print('Output: {0}'.format(args.output))

    photos = []
    with open(args.input, 'r') as in_file:
        total_photos = int(in_file.readline())
        for i in range(total_photos):
            photo_args = in_file.readline().split()
            photo_tags = set(photo_args[2:])
            photo = Photo(id=[i],
                          is_vertical=photo_args[0] == 'V',
                          num_tags=int(photo_args[1]),
                          tags=photo_tags)
            photos.append(photo)

    solution = solve(photos)

    with open(args.output, 'w') as f:
        f.write('{0}\n'.format(len(solution)))
        for photo in solution:
            f.write('{0}\n'.format(' '.join(map(str, photo.id))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input filename', required=True)
    parser.add_argument('--output', type=str, help='Output filename', required=True)
    main(args=parser.parse_args())
