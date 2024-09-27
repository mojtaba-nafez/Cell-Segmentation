# Copyright 2021 The AI-Medic\Cell-Classification Authors. All Rights Reserved.
# License stuff will be written here later...

"""UNet Model (with Batch normalization) implemented from scratch using tensorflow.keras

UNet is an architecture that has many use cases in image segmentation tasks.

It supports various image sizes starting from 32*32

In the architecture; an attention gate is also used and can be applied in the model
by the user (by setting attention=True when passing parameters)

In the table below you can see multiple performances resulted from different configurations of UNet:

+--------------------------+---------------------+---------------------+--------------+
| Model Name               |   DSB-2018 Val Dice |   DSB-2018 Val Loss | Params (M)   |
+==========================+=====================+=====================+==============+
| UNet-pretrain            |               91.30 |              0.1672 | 24,157,377   |
+--------------------------+---------------------+---------------------+--------------+

* DSB-2018 : DATA-SCIENCE-BOWL-2018

Reference:
  - [Github name](https://github.com/yingkaisha/keras-unet-collection)
"""

import os
import sys

sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
from tensorflow.keras.models import Model
from keras_unet_collection import models as models_collection
from params import get_args


class UNet:
    """
          UNet class
          Instantiates the UNet architecture.

          Reference:
                - [Github name](https://github.com/yingkaisha/keras-unet-collection)

          For image segmentation use cases, see
              [this page for detailed examples](
                https://keras.io/examples/vision/oxford_pets_image_segmentation/)
       """

    def __init__(self, input_shape=(256, 256, 3), **kwargs):
        """
        Parameters
        ----------
        input_shape: shape tuple, in "channels_last" format;
           it should have exactly 3 inputs channels, and width and
           height should be no smaller than 32.
           E.g. `(256, 256, 3)` would be one valid value. Default to `None`.
        """
        self.input_shape = input_shape

    def get_model(self) -> Model:
        """
        This method returns a Keras image segmentation model.

        Returns
        -------
        A `Tensorflow.keras.Model` instance.
        """
        args = get_args('unet_pretrain')
        model = models_collection.unet_2d(input_size=self.input_shape,
                                          filter_num=[64, 128, 256, 512, 1024],
                                          n_labels=1,
                                          stack_num_down=2,
                                          stack_num_up=2,
                                          activation="ReLU",
                                          output_activation="Sigmoid",
                                          batch_norm=True,
                                          pool=True,
                                          unpool=True,
                                          backbone=args.backbone,
                                          weights="imagenet",
                                          freeze_backbone=args.freeze_backbone,
                                          freeze_batch_norm=False,
                                          name="pretrain_unet")
        return model
