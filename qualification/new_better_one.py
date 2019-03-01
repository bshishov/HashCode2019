import argparse
from typing import List
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Photo:
    id: List[int]
    is_vertical: bool
    num_tags: int
    tags: set


def merge(v1: Photo, v2: Photo) -> Photo:
    assert v1.is_vertical == v2.is_vertical == True
    merged_tags = v1.tags.union(v2.tags)
    return Photo(v1.id + v2.id, False, len(merged_tags), merged_tags)


def solve(photos: List[Photo]) -> List[Photo]:
    horizontal = []
    vertical = []
    for photo in photos:
        if photo.is_vertical:
            vertical.append(photo)
        else:
            horizontal.append(photo)

    progress_bar = tqdm(total=len(vertical), desc='Vertical merging')
    for iteration in range(len(vertical)):
        progress_bar.update()
        if len(vertical) <= 1:
            break

        v1 = vertical.pop(0)
        max_h = -100000
        max_h_idx = None
        for i, v2 in enumerate(vertical):
            # Vertical merge heuristic
            h = - len(v1.tags.intersection(v2.tags))
            if h > max_h:
                max_h_idx = i
                max_h = h

        if max_h_idx is not None:
            v2 = vertical.pop(max_h_idx)
            horizontal.append(merge(v1, v2))
    progress_bar.close()

    photos = sorted(horizontal, key=lambda x: x.num_tags)
    current_photo = photos.pop(0)
    results = [current_photo]

    for _ in tqdm(range(len(photos)), desc='Arranging slides'):
        if len(photos) == 0:
            break

        max_score = -100000
        max_score_idx = None
        for i, photo2 in enumerate(photos):
            score = min(len(current_photo.tags.difference(photo2.tags)),
                        len(current_photo.tags.intersection(photo2.tags)),
                        len(photo2.tags.difference(current_photo.tags)))
            if score > max_score:
                max_score_idx = i
                max_score = score

        if max_score == 0:
            current_photo = photos.pop(0)
            results.append(current_photo)
            continue

        photo2 = photos.pop(max_score_idx)
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
