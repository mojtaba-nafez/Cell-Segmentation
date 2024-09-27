# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""
This module contains a function to set program to use gpu if one is available.
"""

def get_gpu_grower():
    """
    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
