# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""DeepLabV3PlusAttention(using Resnet 50 as backbone) model implemented from scratch using tensorflow.keras

DeepLabV3Plus is a new architecture and has many use cases in image segmentation tasks.
also attention is one of the novel method to increase your model performance.

It supports various image sizes starting from 32*32

Encoder part of architecture is a Resnet 50, pretarin on imagenet.

In the table below you can see multiple performances resulted from different configurations
of UNet:

+------------------------+---------------------+---------------------+--------------+
| Model Name             |   DSB-2018 Val Dice |   DSB-2018 Val Loss | Params (M)   |
+========================+=====================+=====================+==============+
| DeepLabV3PlusAttention |               90.62 |              0.1825 | 12,348,535   |
+------------------------+---------------------+---------------------+--------------+

* DSB-2018 : DATA-SCIENCE-BOWL-2018

Reference:
  - [code](https://keras.io/examples/vision/deeplabv3_plus/)
  - [paper](https://sh-tsang.medium.com/review-deeplabv3-atrous-separable-convolution-semantic-segmentation-a625f6e83b90)
"""
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid

from tensorflow.keras.models import Model
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple, Union
from utils.attention import *


class DeepLabV3PlusAttention:
    """
    DeepLabV3Plus class
    Instantiates the DeepLabV3Plus architecture.

    Reference:
        - [Github source](https://keras.io/examples/vision/deeplabv3_plus/)

    For image segmentation use cases, see
      [this page for detailed examples](
        https://keras.io/examples/vision/deeplabv3_plus/)
    """

    def __init__(self, input_shape: Tuple[Union[None, int], Union[None, int], int] = (None, None, 3), **kwargs):
        """
        Parameters
        ----------
        input_shape: shape tuple, in "channels_last" format;
            it should have exactly 3 inputs channels, and width and
            height should be no smaller than 32.
            E.g. `(256, 256, 3)` would be one valid value. Default to `None`.
        resnet_trainable: flag type
            set resnet(backbone of model) to non-trainable parameter.
            [True, False]
        layer_trainable_trainable: flag type
            set all parameter except resnet parameters to non-trainable parameter.
            [True, False]
        """
        self.input_shape = input_shape
        self.resnet_trainable = kwargs.get("resnet_trainable", None)
        self.layer_trainable_trainable = kwargs.get("layer_trainable_trainable", None)

    def get_model(self) -> Model:
        """
        This method returns a Keras image segmentation model.

        Returns
        -------
        A `Tensorflow.keras.Model` instance.
        """

        def convolution_block(
                block_input,
                num_filters=256,
                kernel_size=3,
                dilation_rate=1,
                padding="same",
                use_bias=False,
        ):
            """
            This method returns a Keras convolution block.
            you can use it to build a model.

            Parameters
            --------
            block_input:    KerasTensor, input layer of block
            num_filters:    int, number of conv filter
            kernel_size:    int, size of conv kernel
            dilation_rate:  int, Atrous convolution rate (https://stackoverflow.com/questions/63073760/using-dilated-convolution-in-keras)
            padding:        int, padding size of conv
            use_bias:       flag, use bias or not


            Returns
            -------
            A `KerasTensor` instance.
            """
            x = layers.Conv2D(
                num_filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                padding=padding,
                use_bias=use_bias,
                kernel_initializer=tf.keras.initializers.he_normal(),
                trainable=self.layer_trainable_trainable,
            )(block_input)
            x = layers.BatchNormalization(trainable=self.layer_trainable_trainable)(x)
            return tf.nn.relu(x)

        def DilatedSpatialPyramidPooling(dspp_input):
            """
            This method returns a Keras convolution block.
            you can use it to build a model.

            Parameters
            -------
            dspp_input: KerasTensor, Dilated Spatial Pyramid Pooling

            Returns
            -------
            A `KerasTensor` instance.
            """
            dims = dspp_input.shape
            x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
            x = convolution_block(x, kernel_size=1, use_bias=True)
            out_pool = layers.UpSampling2D(
                size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
                trainable=self.layer_trainable_trainable
            )(x)
            out_pool = cbam_block(out_pool)
            out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
            out_1 = cbam_block(out_1)
            out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
            out_6 = cbam_block(out_6)
            out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
            out_12 = cbam_block(out_12)
            out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)
            out_18 = cbam_block(out_18)
            x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
            x = cbam_block(x)
            output = convolution_block(x, kernel_size=1)

            return output

        model_input = keras.Input(self.input_shape)
        image_weight = self.input_shape[0]
        image_height = self.input_shape[1]

        resnet50 = keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_tensor=model_input
        )
        resnet50.trainable = self.resnet_trainable
        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = DilatedSpatialPyramidPooling(x)

        input_a = layers.UpSampling2D(
            size=(image_weight // 4 // x.shape[1], image_height // 4 // x.shape[2]),
            interpolation="bilinear",
        )(x)
        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = cbam_block(input_b)

        input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = convolution_block(x)
        x = convolution_block(x)
        x = layers.UpSampling2D(
            size=(image_weight // x.shape[1], image_height // x.shape[2]),
            interpolation="bilinear",
        )(x)
        model_output = layers.Conv2D(1, kernel_size=(1, 1), padding="same", activation='sigmoid')(x)
        return keras.Model(inputs=model_input, outputs=model_output)
