import os
import argparse
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

from lmdb_connector import LmdbConnector


def check_image_is_valid(img_bin):
    if img_bin is None:
        return False
    img_buf = np.frombuffer(img_bin, dtype=np.uint8)
    img = cv2.imdecode(img_buf, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[0], img.shape[1]
    if height * width == 0:
        return False
    return True


def process_func(args):
    i, raw_line = args
    image_path, label = raw_line.strip('\n').split('\t')
    image_path = os.path.join(process_func.input_dir, image_path)

    if not os.path.exists(image_path):
        print('%s does not exist' % image_path)
        return None

    with open(image_path, 'rb') as f:
        img_bin = f.read()

    if process_func.check_valid:
        try:
            if not check_image_is_valid(img_bin):
                print('%s is not a valid image' % image_path)
                return None
        except:
            print('error occured', i)
            with open(process_func.output_dir + '/error_image_log.txt', 'a') as log:
                log.write('%s-th image data occured error\n' % str(i))
            return None

    return (img_bin, label)


def init_process_func(input_dir, output_dir, check_valid):
    setattr(process_func, 'input_dir', input_dir)
    setattr(process_func, 'output_dir', output_dir)
    setattr(process_func, 'check_valid', check_valid)


def prepare_lmdb(input_dir, gt_file, output_dir, check_valid, workers):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        input_dir  : input folder path where starts image_path
        output_dir : LMDB output path
        gt_file     : list of image path and label
        check_valid : if true, check the validity of every image
    """
    os.makedirs(output_dir, exist_ok=True)
    connector = LmdbConnector(output_dir, mode='w')

    with open(gt_file, 'r', encoding='utf-8') as data:
        datalist = data.readlines()
    n_samples = len(datalist)

    jobs = []
    for i in range(n_samples):
        jobs.append((i, datalist[i]))

    pool = Pool(workers, init_process_func, (input_dir, output_dir, check_valid))
    multiprocessing_generator = tqdm(
        pool.imap(process_func, jobs),
        total=n_samples,
    )
    connector.write(multiprocessing_generator)
    pool.terminate()
    print('Created dataset with %d samples' % n_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='input folder path where starts image_path')
    parser.add_argument('--output_dir', required=True, help='LMDB output path')
    parser.add_argument('--gt_file', required=True, help='list of image path and label')
    parser.add_argument('--workers', type=int, default=4, help='number of cpu workers')
    parser.add_argument('--skip_check', action='store_true', help='skip checking the validity of every image')
    opt = parser.parse_args()

    prepare_lmdb(opt.input_dir, opt.gt_file, opt.output_dir, not opt.skip_check, opt.workers)
