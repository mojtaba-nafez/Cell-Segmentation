# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""
Preprocessing module
    Download && extract zip file of dataset

This modules contain following Functions:
    -   preprocess
"""

import os
from os.path import join
from glob import glob
import numpy as np
from argparse import ArgumentParser
import shutil
import requests
from deep_utils import remove_create
from tqdm import tqdm
import cv2


def get_paths(dir_):
    """

    Parameters
    ----------
    dir_        string , data path (to read)

    Returns
    -------
    (x, y)      (1D array of image path, 1D array of mask path), (images, masks)
    """
    x, y = [], []
    for path in os.listdir(dir_):
        f_path = join(dir_, path)
        images = join(f_path, 'images')
        masks = join(f_path, 'masks')
        x.append(images)
        y.append(masks)
    return x, y


def load_data_path(train_dir='../dataset/train/', test_dir='../dataset/test/', val_dir='../dataset/val/'):
    """
    Parameters
    ----------
    train_dir   train data path
    test_dir    test data path
    val_dir     validation data path

    Returns
    -------
    ((train_x, train_y), (val_x, val_y), test_x)
                ((1D array of train image path, 1D array of train mask path),
                (1D array of validation image path, 1D ,array of validation mask path),
                 1D array of test data image path))
    """
    train_x, train_y = get_paths(train_dir)
    val_x, val_y = get_paths(val_dir)
    test_x, _ = get_paths(test_dir)

    return (train_x, train_y), (val_x, val_y), test_x


def preprocess(
        test_url='https://raw.githubusercontent.com/kamalkraj/DATA-SCIENCE-BOWL-2018/master/data/stage1_test.zip',
        train_url='https://raw.githubusercontent.com/kamalkraj/DATA-SCIENCE-BOWL-2018/master/data/stage1_train.zip',
        test_filename='stage1_test.zip',
        train_filename='stage1_train.zip',
        train_dir='../dataset/train/',
        test_dir='../dataset/test/',
        val_dir='../dataset/val/',
        combine=True,
        remove_zips=False,
):
    """
    Downloads && extracts dataset zip files

    Parameters
    ----------
    test_url        str,
    train_url       str,
    test_filename   str,
    train_filename  str,
    train_dir       str,
    test_dir        str,
    val_dir         str,
    combine         bool, Whether combine different mask and org images
    remove_zips     bool, Whether remove downloaded zipfiles

    """
    remove_create(train_dir)
    remove_create(test_dir)
    remove_create(val_dir)

    if not os.path.exists(test_filename):
        print(f'Downloading {test_url}')
        r = requests.get(test_url, allow_redirects=True)
        open(test_filename, 'wb').write(r.content)
        print(f"Downloading is Done")
    else:
        print(f'{test_filename} Already exists')
    if not os.path.exists(train_filename):
        print(f"Downloading {train_url}")
        r = requests.get(train_url, allow_redirects=True)
        open(train_filename, 'wb').write(r.content)
        print(f"Downloading is Done")
    else:
        print(f'{train_filename} Already exists')

    os.system("unzip ./" + train_filename + " -d " + train_dir)
    os.system("unzip ./" + test_filename + " -d " + test_dir)

    data_path = np.array(glob(train_dir + "*"))
    val_path = data_path[:int(0.2 * len(data_path))]
    for source in val_path:
        shutil.move(source, val_dir)
    if remove_zips:
        os.remove(test_filename)
        os.remove(train_filename)
    print("Preprocess is done!")

    # Combine masks & images
    if combine:
        (train_x, train_y), (val_x, val_y), test_x = load_data_path(train_dir, test_dir, val_dir)
        combine_mask_images(train_x, train_y)
        combine_mask_images(val_x, val_y)
        combine_mask_images(test_x)


def combine_mask_images(x, y=None):
    print('Combining masks')
    if y is None:
        for img_path in tqdm(x, total=len(x), desc="combining"):
            img = np.sum(np.stack(
                [cv2.imread(join(img_path, img)) for img in os.listdir(img_path)],
                axis=3), axis=3)
            cv2.imwrite(img_path + ".png", img)
    else:
        for img_path, mask_path in tqdm(zip(x, y), total=len(x), desc="combining"):
            mask_img = np.sum(np.stack(
                [cv2.imread(join(mask_path, mask), cv2.IMREAD_GRAYSCALE) for mask in os.listdir(mask_path)],
                axis=2), axis=2, keepdims=True)
            cv2.imwrite(mask_path + ".png", mask_img)
            img = np.sum(np.stack(
                [cv2.imread(join(img_path, img)) for img in os.listdir(img_path)],
                axis=3), axis=3)
            cv2.imwrite(img_path + ".png", img)
    print('Combining is done')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--test_url', type=str,
                        default='https://raw.githubusercontent.com/kamalkraj/DATA-SCIENCE-BOWL-2018/master/data/stage1_test.zip',
                        help='test data url')
    parser.add_argument('--train_url', type=str,
                        default='https://raw.githubusercontent.com/kamalkraj/DATA-SCIENCE-BOWL-2018/master/data/stage1_train.zip',
                        help='train data url')
    parser.add_argument('--train_dir', type=str, default='../dataset/train/', help='train directory')
    parser.add_argument('--test_dir', type=str, default='../dataset/test/', help='test directory')
    parser.add_argument('--val_dir', type=str, default='../dataset/val/', help='validation directory')
    parser.add_argument('--train_filename', type=str, default='stage1_train.zip', help='train zip file name')
    parser.add_argument('--test_filename', type=str, default='stage1_test.zip', help='test zip file name')
    parser.add_argument('--combine', action='store_true',
                        help="combine masks")
    parser.add_argument('--not-combine', dest='combine', action='store_false',
                        help="Don't combine masks")
    parser.set_defaults(combine=True)
    parser.add_argument('--remove-zips', action='store_true',
                        help="remove-zips")
    parser.add_argument('--not-remove-zips', dest='remove_zips', action='store_false',
                        help="Don't remove zips")
    parser.set_defaults(remove_zips=False)
    args = parser.parse_args()
    print(args)
    preprocess(test_url=args.test_url,
               train_url=args.train_url,
               test_filename=args.test_filename,
               train_filename=args.train_filename,
               train_dir=args.train_dir,
               test_dir=args.test_dir,
               val_dir=args.val_dir,
               remove_zips=args.remove_zips,
               combine=args.combine
               )
