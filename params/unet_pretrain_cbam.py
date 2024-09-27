# Copyright 2021 The AI-Medic\Cell-Classification Authors. All Rights Reserved.
# License stuff will be written here later...

"""
Set unet parameters before execution
"""

from .main import main_args


def unet_pretrain_cbam_args():
    """
    Parameters
    ----------
    None

    Returns
    -------
    Returns hyper-parameters of unet model to train.
    """
    parser = main_args()
    parser.add_argument('--pretrain', type=str, default=None, help='path of pretrain .h5 wights')
    parser.add_argument('--input_shape', type=list, default=[256, 256, 3], help='use attention module: True or False')
    parser.add_argument('--resnet_trainable', dest='resnet_trainable', action='store_true',
                        help="backbone of deeplabv3 is pretrain resnet (you want to train or not?)")
    parser.set_defaults(resnet_trainable=False)
    return parser.parse_args()
