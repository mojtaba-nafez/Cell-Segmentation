# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""
Save Plots in experiment tracking (mlflow)

this modules contain following Functions:
    -   iou_coef
    -   dice_coef
    -   dice_coef_loss
    -   bce_dice_loss
"""

import matplotlib.pyplot as plt
import numpy as np
from .mlflow_handler import MLFlowHandler


def get_plots(model, val_loader, mlflow_handler: MLFlowHandler):
    """

    Parameters
    ----------
    model           A `Tensorflow.keras.Model` instance.
    val_loader      DataGenerator
    mlflow_handler  a class, an interface to work easily with mlfow

    Returns
    -------
    Nothing         just save plots of predicted mask by model to visual intuition
    """
    x, y = val_loader.__getitem__(2)
    y_pre = model.predict(x)
    if len(y[0].shape)>2 and y[0].shape[2]>1:
        y_pre=y_pre[:,:,:,1]
        y=y[:,:,:,1]
    for i in range(x.shape[0]):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.tight_layout()
        ax1.imshow(x[i])
        ax1.set_title('input')
        ax2.imshow(y[i])
        ax2.set_title('mask')
        ax3.imshow(np.squeeze(y_pre[i]))
        ax3.set_title('predicted')
        mlflow_handler.add_figure(fig, 'images/predicted_images/' + str(i) + '.png')
