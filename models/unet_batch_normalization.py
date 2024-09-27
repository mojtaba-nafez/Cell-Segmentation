# Copyright 2021 The AI-Medic\Cell-Classification Authors. All Rights Reserved.
# License stuff will be written here later...

"""UNet Model (with Batch normalization) implemented from scratch using tensorflow.keras

UNet is an architecture that has many use cases in image segmentation tasks.

It supports various image sizes starting from 32*32

In the architecture; an attention gate is also used and can be applied in the model
by the user (by setting attention=True when passing parameters)

In the table below you can see multiple performances resulted from different configurations
of UNet:

+--------------------------+---------------------+---------------------+--------------+-------------+
| Model Name               |   DSB-2018 Val Dice |   DSB-2018 Val Loss | Params (M)   | Attention   |
+==========================+=====================+=====================+==============+=============+
| UNet-Batch-Normalization |                89.6 |              0.1725 | 34,614,369   | None        |
+--------------------------+---------------------+---------------------+--------------+-------------+
| UNet-Batch-Normalization |                88.2 |              0.202  | 42,656,518   | Att_Gate    |
+--------------------------+---------------------+---------------------+--------------+-------------+

* DSB-2018 : DATA-SCIENCE-BOWL-2018


Reference:
  - [Github name](https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/tree/a4150d2d68b73ea5682334b976707a5e21fa043e/model)
"""

import os
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, \
    BatchNormalization
from typing import Tuple, Union

sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
from utils.attention import AttnGatingBlock as attention
from params import get_args


class UNet:
    """
    UNet class
    Instantiates the UNet architecture.

    Reference:
          - [Github name](https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/tree/a4150d2d68b73ea5682334b976707a5e21fa043e/model)

    For image segmentation use cases, see
        [this page for detailed examples](
          https://keras.io/examples/vision/oxford_pets_image_segmentation/)
    """

    def __init__(self, input_shape: Tuple[Union[None, int], Union[None, int], int] = (None, None, 3), **kwargs):
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
        inputs = Input(shape=self.input_shape)

        down0 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
        down0 = BatchNormalization()(down0)
        down0 = Activation('relu')(down0)
        down0 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(down0)
        down0 = BatchNormalization()(down0)
        down0 = Activation('relu')(down0)
        down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
        # 128

        down1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(down0_pool)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(down1)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
        # 64

        down2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(down1_pool)
        down2 = BatchNormalization()(down2)
        down2 = Activation('relu')(down2)
        down2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(down2)
        down2 = BatchNormalization()(down2)
        down2 = Activation('relu')(down2)
        down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
        # 32

        down3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(down2_pool)
        down3 = BatchNormalization()(down3)
        down3 = Activation('relu')(down3)
        down3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(down3)
        down3 = BatchNormalization()(down3)
        down3 = Activation('relu')(down3)
        down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
        # 16

        down4 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(down3_pool)
        down4 = BatchNormalization()(down4)
        down4 = Activation('relu')(down4)
        down4 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(down4)
        down4 = BatchNormalization()(down4)
        down4 = Activation('relu')(down4)
        down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
        # 8

        center = Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal')(down4_pool)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        center = Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal')(center)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        # center

        args = get_args('unet_bn')
        if args.attention:
            attention4 = attention(down4, center, 1024)
            attention3 = attention(down3, down4, 512)
            attention2 = attention(down2, down3, 256)
            attention1 = attention(down1, down2, 128)
            attention0 = attention(down0, down1, 64)

            down4 = attention4
            down3 = attention3
            down2 = attention2
            down1 = attention1
            down0 = attention0

        up4 = UpSampling2D((2, 2))(center)
        up4 = concatenate([down4, up4], axis=3)
        up4 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(up4)
        up4 = BatchNormalization()(up4)
        up4 = Activation('relu')(up4)
        up4 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(up4)
        up4 = BatchNormalization()(up4)
        up4 = Activation('relu')(up4)
        up4 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(up4)
        up4 = BatchNormalization()(up4)
        up4 = Activation('relu')(up4)
        # 16

        up3 = UpSampling2D((2, 2))(up4)
        up3 = concatenate([down3, up3], axis=3)
        up3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(up3)
        up3 = BatchNormalization()(up3)
        up3 = Activation('relu')(up3)
        up3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(up3)
        up3 = BatchNormalization()(up3)
        up3 = Activation('relu')(up3)
        up3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(up3)
        up3 = BatchNormalization()(up3)
        up3 = Activation('relu')(up3)
        # 32

        up2 = UpSampling2D((2, 2))(up3)
        up2 = concatenate([down2, up2], axis=3)
        up2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        up2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        up2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        # 64

        up1 = UpSampling2D((2, 2))(up2)
        up1 = concatenate([down1, up1], axis=3)
        up1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        # 128

        up0 = UpSampling2D((2, 2))(up1)
        up0 = concatenate([down0, up0], axis=3)
        up0 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(up0)
        up0 = BatchNormalization()(up0)
        up0 = Activation('relu')(up0)
        up0 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(up0)
        up0 = BatchNormalization()(up0)
        up0 = Activation('relu')(up0)
        up0 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(up0)
        up0 = BatchNormalization()(up0)
        up0 = Activation('relu')(up0)

        classify = Conv2D(1, (1, 1), activation='sigmoid')(up0)  # using dataGen means 1,3,4 channels only
        model = Model(inputs=inputs, outputs=classify)
        model.summary()
        return model
