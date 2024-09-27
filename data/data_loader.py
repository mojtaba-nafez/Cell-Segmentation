# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""
Preprocessing && Data_Loader module
    Prepare 3 Generator to use in training procedure

This module contains the following functions:
    -   load_data_path
    -   get_paths
    -   DataGenerator
    -   get_loader
    -   test_generator
"""

import os
from os.path import join
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import math
import cv2
from models import load_model
import albumentations as A
from tensorflow.keras.utils import to_categorical


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
        images = join(f_path, 'images.png')
        masks = join(f_path, 'masks.png')
        x.append(images)
        y.append(masks)
    return x, y


class DataGenerator(tf.keras.utils.Sequence):
    """
    Costume DataGenerator class
       Instantiates the tf.keras.utils.Sequence

       Reference:
             - [Github name](https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/tree/a4150d2d68b73ea5682334b976707a5e21fa043e/model)

       For costume data generator use cases, see
           [this page for detailed examples](
             https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence)
       """

    def __init__(self, x_set, y_set=None,
                 batch_size=32,
                 img_size=(256, 256),
                 img_channel=3,
                 augmentation_p=0.5,
                 shuffle=True,
                 categorical=False):
        """

        Parameters
        ----------
        x_set           1D array of path, address of images
        y_set           1D array of path, address of masks
        batch_size      int, batch size of generator
        img_size        (None, None), image size
        img_channel     int, number image channel
        augmentation_p  float ( 0<=aug_p<=1 ), augmentation probability
        shuffle         Flag, shuffle data per epoch or not
        """
        self.img_paths, self.mask_paths = np.array(x_set), np.array(y_set)
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.img_size = img_size
        self.img_channel = img_channel
        self.categorical = categorical
        self.transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=90, p=0.5)
        ], p=augmentation_p)
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        execute end of each epoch (shuffle the data)
        """
        if self.shuffle:
            indices = np.random.permutation(len(self.img_paths)).astype(np.int)
            self.img_paths, self.mask_paths = self.img_paths[indices], self.mask_paths[indices]

    def __len__(self):
        """

        Returns
        -------
        number of batch in sequence
        """
        return math.ceil(len(self.img_paths) / self.batch_size)

    def __getitem__(self, idx):
        """

        Parameters
        ----------
        idx     sequence number

        Returns
        -------
        a batch of data
        """
        batch_img = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.mask_paths is None:
            x = np.array([cv2.resize(cv2.imread(p), self.img_size) for p in batch_img])
            return x / 255
        batch_mask = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = np.zeros((self.batch_size, *self.img_size, self.img_channel), dtype=float)
        y = np.zeros((self.batch_size, *self.img_size), dtype=float)

        for i, (img_path, mask_path) in enumerate(zip(batch_img, batch_mask)):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img.astype(np.uint8), self.img_size)
            mask = cv2.resize(mask.astype(np.uint8), self.img_size, interpolation=cv2.INTER_NEAREST)
            augmented = self.transform(image=img, mask=mask)
            x[i] = augmented['image'].astype(np.uint8)[..., ::-1] # BGR to RGB
            y[i] = augmented['mask'].astype(np.uint8)
        y = y.reshape((self.batch_size, *self.img_size))
        x, y = x / 255, y / 255
        if self.categorical:
            y = to_categorical(y, dtype="float32")
        return x, y


def get_loader(batch_size=64,
               train_dir='../dataset/train/',
               test_dir='../dataset/test/',
               val_dir='../dataset/val/',
               augmentation_p=0.5,
               shuffle=True,
               categorical=False):
    """

    Parameters
    ----------
    batch_size          generator batch size
    train_dir           train data path
    test_dir            test data path
    val_dir             validation data path
    augmentation_p      float( 0 <= aug_p <= 1 ), probability of augmentation
    shuffle             Flag, shuffling data each epoch

    Returns
    -------
    (train_gen, val_gen, test_gen) :    3 Generator for train, validation, test
    """
    (x_train_path, y_train_path), (x_val_path, y_val_path), (x_test_path) = load_data_path(train_dir=train_dir,
                                                                                           test_dir=test_dir,
                                                                                           val_dir=val_dir)
    print('train data count: ', len(x_train_path))
    print('val data count: ', len(x_val_path))
    print('test data count: ', len(x_test_path))

    train_gen = DataGenerator(x_set=x_train_path, y_set=y_train_path, batch_size=batch_size,
                              augmentation_p=augmentation_p, shuffle=shuffle, categorical=categorical)
    val_gen = DataGenerator(x_set=x_val_path, y_set=y_val_path, batch_size=batch_size, augmentation_p=augmentation_p,
                            shuffle=shuffle, categorical=categorical)
    test_gen = DataGenerator(x_set=x_val_path, batch_size=batch_size, augmentation_p=0, shuffle=False)
    return train_gen, val_gen, test_gen


def test_generator(generator, test=0, name='', bs=2):
    """

    Parameters
    ----------
    generator
    test
    name
    bs

    Returns
    -------
    nothing   -> just plot some example of data

    """
    if test == 0:
        x, y = generator.__getitem__(1)
    else:
        x = generator.__getitem__(1)

    for i in range(bs):
        if test != 0:
            plt.imshow(x[i])
            plt.title(name)
            plt.show()
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(x[i])
        ax1.set_title(name + ' image ')
        ax2.imshow(y[i])
        ax2.set_title(name + ' mask ')
    print('max value of val data: ', np.max(x[0]))
    if test == 0:
        print('max value of training data: ', np.max(y[0]))


if __name__ == '__main__':
    batch_size = 2
    train_gen, val_gen, test_gen = get_loader(batch_size=batch_size)
    model = load_model('unet')
    model.summary()
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['acc'])
    print("start training")
    model.fit(train_gen, batch_size=batch_size, epochs=2)
    x, y = train_gen[12]
    y_pre = model.predict(x)
    plt.imshow(y[0])
    plt.show()
    plt.imshow(np.squeeze(y_pre[0]))
    plt.show()
