# Copyright 2021 The AI-Medic\Cell-Classification Authors. All Rights Reserved.
# License stuff will be written here later...

"""
Set unet_pretrain parameters before execution
"""

from .main import main_args


def unet_pretrain_args():
    """
    Parameters
    ----------
    None

    Returns
    -------
    Returns hyper-parameters of unet_pretrain model to train.
    """
    parser = main_args()
    parser.add_argument('--augmentation', dest='augmentation', action='store_true', help='augmentations')
    parser.add_argument('--no_augmentation', dest='augmentation', action='store_false', help='no augmentations')
    parser.add_argument('--pretrain', type=str, default=None, help='path of pretrain .h5 wights')
    parser.add_argument('--attention', dest='attention', action='store_true', help="model with attention module")
    parser.add_argument('--no_attention', dest='attention', action='store_false', help="model without attention module")
    parser.set_defaults(attention=True, augmentation=False)
    parser.add_argument('--input_shape', type=list, default=[256, 256, 3], help='use attention module: True or False')
    parser.add_argument('--backbone', type=str, default='ResNet50',
                        help='Should be one of the `tensorflow.keras.applications` class. None means no backbone')
    parser.add_argument('--no-freeze-backbone', dest='freeze_backbone', default=False,
                        action='store_false', help="Don't freeze backbone")
    parser.add_argument('--freeze-backbone', dest='freeze_backbone',
                        action='store_true', help='Freeze backbone')

    return parser.parse_args()
