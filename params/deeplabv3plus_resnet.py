# Copyright 2021 The AI-Medic\Cell-Classification Authors. All Rights Reserved.
# License stuff will be written here later...

"""
Set Deeplabv3plus_resnet parameters before execution
"""

from .main import main_args


def deeplabv3p_resnet_args():
    """

    Returns
    -------
    Returns hyper parameter of deeplabv3plus_resnet model to train
    """
    parser = main_args()

    parser.add_argument('--pretrain', type=str, default=None, help='path of pretrain .h5 wights')
    parser.add_argument('--resnet_trainable', dest='resnet_trainable', action='store_true',
                        help="backbone of deeplabv3 is pretrain resnet (you want to train or not?)")
    parser.set_defaults(resnet_trainable=False)
    parser.add_argument('--layer_trainable_trainable', dest='layer_trainable_trainable', action='store_true',
                        help="Freeze all parameter except backbone")
    parser.set_defaults(layer_trainable_trainable=False)
    parser.add_argument('--input_shape', type=list, default=[256, 256, 3], help='use attention module: True or False')

    return parser.parse_args()
