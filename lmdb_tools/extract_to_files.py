import argparse
import os
from multiprocessing import Pool

from tqdm import tqdm

from lmdb_connector import LmdbConnector


def save_labels(connector, path, indexes, img_paths):
    with open(path, 'w') as fw:
        for index, img_path in tqdm(zip(indexes, img_paths), total=len(indexes)):
            label = connector.get_label(index)
            fw.write(f'{img_path}\t{label}\n')


def extract_to_file(args):
    input_lmdb, index, output_dir, path, rgb = args
    connector = LmdbConnector(input_lmdb, mode='r')
    path = os.path.join(output_dir, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = connector.get_image(index, rgb=rgb)
    img.save(path)


def extract_to_files(input_lmdb, output_dir, rgb, thread_count):
    os.makedirs(output_dir, exist_ok=True)

    connector = LmdbConnector(input_lmdb, mode='r')
    indexes = list(connector.indexes_generator())
    img_paths = [os.path.join('{:04}'.format(index // 10000), '{:08}.jpg'.format(index))
                 for index in indexes]

    print('start extracting labels ...')
    path_labels = os.path.join(output_dir, 'labels.txt')
    save_labels(connector, path_labels, indexes, img_paths)

    total = connector.count()

    del connector

    print('start extracting images ...')
    p = Pool(thread_count)
    for _ in tqdm(
        p.imap_unordered(
            func=extract_to_file,
            iterable=zip(
                [input_lmdb] * total,
                indexes,
                [output_dir] * total,
                img_paths,
                [rgb] * total
            )
        ),
        total=total,
    ):
        pass
    p.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_lmdb', required=True, help='input lmdb folder')
    parser.add_argument('--output_dir', required=True, help='output folder')
    parser.add_argument('--workers', type=int, default=4, help='number of cpu workers')
    parser.add_argument('--rgb', action='store_true', help='rgb')
    opt = parser.parse_args()

    extract_to_files(opt.input_lmdb, opt.output_dir, opt.rgb, opt.workers)
