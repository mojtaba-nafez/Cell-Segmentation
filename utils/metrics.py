# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""
We define performance metrics && new loss function in this module.

this modules contain following Functions:
    -   iou_coef
    -   dice_coef
    -   dice_coef_loss
    -   bce_dice_loss
"""

import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy


def iou_coef(y_true, y_pred, smooth=1):
    """
    Parameters
    ----------
    y_true  (batch_size, None, None, channel), true mask of image (true labeled)
    y_pred  (batch_size, None, None, channel) predicted mask of image by model
    smooth  int, handle zero denominator in division

    Returns
    -------
    iou     float, performance metric for semantic segmentation

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth))
    return iou


def dice_coef(y_true, y_pred, smooth=1.):
    """
    Parameters
    ----------
    y_true  (batch_size, None, None, channel), true mask of image (true labeled)
    y_pred  (batch_size, None, None, channel) predicted mask of image by model
    smooth  int, handle zero denominator in division

    Returns
    -------
    dice_coef     float, performance metric for semantic segmentation

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f)
    score = (2. * intersection + smooth) / (union + smooth)
    return score


def dice_coef_loss(y_true, y_pred):
    """
    Parameters
    ----------
    y_true  (batch_size, None, None, channel), true mask of image (true labeled)
    y_pred  (batch_size, None, None, channel) predicted mask of image by model

    Returns
    -------
    dice_coef_loss     float, performance metric for semantic segmentation, new loss funtion

    """
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """
    Parameters
    ----------
    y_true  (batch_size, None, None, channel), true mask of image (true labeled)
    y_pred  (batch_size, None, None, channel) predicted mask of image by model

    Returns
    -------
    dice_coef_loss     float, performance metric for semantic segmentation, new loss funtion

    """
    loss = binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)
    return loss

def cat_dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def cat_dice_coef_loss(y_true, y_pred):
    """
    Parameters
    ----------
    y_true  (batch_size, None, None, channel), true mask of image (true labeled)
    y_pred  (batch_size, None, None, channel) predicted mask of image by model

    Returns
    -------
    dice_coef_loss     float, performance metric for semantic segmentation, new loss funtion

    """
    return 1 - cat_dice_coef(y_true, y_pred)
def cce_dice_loss(y_true, y_pred):
    """
    Parameters
    ----------
    y_true  (batch_size, None, None, channel), true mask of image (true labeled)
    y_pred  (batch_size, None, None, channel) predicted mask of image by model

    Returns
    -------
    dice_coef_loss     float, performance metric for semantic segmentation, new loss funtion

    """
    loss = (categorical_crossentropy(y_true, y_pred)) + cat_dice_coef_loss(y_true, y_pred)
    return loss