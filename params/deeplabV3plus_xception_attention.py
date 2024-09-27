# Copyright 2021 The AI-Medic\Cell-Classification Authors. All Rights Reserved.
# License stuff will be written here later...

"""
Set Deeplabv3plus_xception parameters before execution
"""

from .main import main_args


def deeplabV3plus_xception_attention_args():
    """

    Returns
    -------
    Returns hyper parameter of deeplabv3plus_xception model to train
    """
    parser = main_args()

    parser.add_argument('--pretrain', type=str, default=None, help='path of pretrain .h5 wights')
    parser.add_argument('--input_shape', type=list, default=[256, 256, 3], help='use attention module: True or False')

    return parser.parse_args()
