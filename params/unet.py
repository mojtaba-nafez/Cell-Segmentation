# Copyright 2021 The AI-Medic\Cell-Classification Authors. All Rights Reserved.
# License stuff will be written here later...

"""
Set unet parameters before execution
"""

from .main import main_args


def unet_args():
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
    parser.add_argument('--attention', dest='attention', action='store_true', help="model with attention module")
    parser.add_argument('--no_attention', dest='attention', action='store_false', help="model without attention module")
    parser.set_defaults(attention=True)
    parser.add_argument('--input_shape', type=list, default=[256, 256, 3], help='use attention module: True or False')
    return parser.parse_args()
