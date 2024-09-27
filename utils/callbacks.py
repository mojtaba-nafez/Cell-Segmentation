# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""
This module contains a function to provide callbacks.
"""

from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from utils.warmup_onecycle import WarmUpCosineDecayScheduler
from datetime import datetime


def get_callbacks(model_path,
                  early_stopping_p,
                  tb_log_dir,
                  save_weights_only,
                  plateau_min_lr,
                  epochs,
                  warmup_epoch,
                  batch_size,
                  learning_rate_base,
                  sample_count,
                  model_name,
                  **kwargs):
    """
    This function use some callbacks from tensorflow.python.keras.callbacks

    Parameters
    ----------
    model_path: str ; path to save the model file.
    early_stopping_p: int ; number of epochs with no improvement after which training will be stopped.
    save_weights_only: bool ; if True, then only the model's weights will be saved.
    plateau_min_lr: float ; lower bound on the learning rate; it is for tensorflow.python.keras.callbacks.ReduceLROnPlateau module
    **kwargs

    Returns
    -------
    checkpoint: a tensorflow.python.keras.callbacks.ModelCheckpoint instance
    reduce_lr: a tensorflow.python.keras.callbacks.ReduceLROnPlateau instance
    early_stopping: a tensorflow.python.keras.callbacks.EarlyStopping instance
    """
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=save_weights_only,
                                 )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,  # new_lr = lr * factor
                                  patience=5,  # number of epochs with no improvment
                                  min_lr=plateau_min_lr,  # lower bound on the learning rate
                                  mode='min',
                                  verbose=1
                                  )

    total_steps = int(epochs * sample_count / batch_size)
    # Compute the number of warmup batches.
    warmup_steps = int(warmup_epoch * sample_count / batch_size)

    # Generate dummy data.
    # warmup_batches = warmup_epoch * sample_count / batch_size

    # Create the Learning rate scheduler.
    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=0.0,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=0, 
                                            verbose=0)

    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_p, verbose=1)

    tensorboard = TensorBoard(log_dir=tb_log_dir + '/' + model_name + '_{}'.format(str(datetime.now()).replace(':', '_').replace(' ','_')),
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    return checkpoint, warm_up_lr, early_stopping, tensorboard, reduce_lr
