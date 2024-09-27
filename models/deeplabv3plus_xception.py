# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""DeepLabV3Plus(using Xception as backbone) model implemented from scratch using tensorflow.keras

DeepLabV3Plus is a new architecture and has many use cases in image segmentation tasks.

It supports various image sizes starting from 32*32

Encoder part of architecture is a Xception (not pretrain)

In the table below you can see multiple performances resulted from different configurations
of UNet:

+------------------------+---------------------+---------------------+--------------+
| Model Name             |   DSB-2018 Val Dice |   DSB-2018 Val Loss | Params (M)   |
+========================+=====================+=====================+==============+
| DeepLabV3Plus_Xception |               89.78 |              0.3061 | 41,016,185   |
+------------------------+---------------------+---------------------+--------------+

* DSB-2018 : DATA-SCIENCE-BOWL-2018

Reference:
  - [code](https://github.com/puruBHU/DeepLabv3plus-keras/)
  - [paper](https://sh-tsang.medium.com/review-deeplabv3-atrous-separable-convolution-semantic-segmentation-a625f6e83b90)
"""

import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, GlobalAveragePooling2D, Reshape, Concatenate
from typing import Tuple, Union
from .xception import *

if K.image_data_format() == 'channels_last':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3


class DeepLabV3Plus:
    """
    DeepLabV3Plus class
    Instantiates the DeepLabV3Plus architecture.

    Reference:
        - [Github source](https://github.com/puruBHU/DeepLabv3plus-keras/)

    For image segmentation use cases, see
      [this page for detailed examples](
        https://keras.io/examples/vision/deeplabv3_plus/)
    """
    def __init__(self, input_shape: Tuple[Union[None, int], Union[None, int], int] = (None, None, 3), backbone='xception', OS=16, **kwargs):
        """
        Parameters
        ----------
        input_shape: shape tuple, in "channels_last" format;
            it should have exactly 3 inputs channels, and width and
            height should be no smaller than 32.
            E.g. `(256, 256, 3)` would be one valid value. Default to `None`.
        backbone: string
            it should have a value from this set: ['xception', 'resnet50',  'resnet101', 'mobilenetv2']
        OS: int
            output stride
            commonly take value 8, 16
        num_classes: int
            in segmentation determine number of label in each pixel (pixel classification)
        """
        self.shape = input_shape
        self.backbone = backbone
        self.OS = OS
        self.num_classes = 1

    def get_model(self) -> Model:
        """
        This method returns a Keras image segmentation model.

        Returns
        -------
        A `Tensorflow.keras.Model` instance.
        """
        def ASPP():
            """
            Function to to perfrom atrous pyramid pooling
            you can use it to build a model.

            Returns
            -------
            A `KerasTensor` instance.
            """

            def f(tensor):
                """
                This method returns a Keras convolution block.
                you can use it to build a model.

                Returns
                -------
                A `KerasTensor` instance.
                """

                # Get shape of the final feature layer of the backbone network
                h, w, c = K.int_shape(tensor)[1:]

                global_avg_pool = GlobalAveragePooling2D(name='global_avg_pool')(tensor)

                # Get the number of output channels from the previous layer
                c = K.int_shape(global_avg_pool)[-1]

                image_level_feature = Reshape((1, 1, c), name='reshape')(global_avg_pool)
                image_level_feature = conv_bn_relu(filters=256, kernel_size=(1, 1), name='bottleneck_GAP')(
                    image_level_feature)
                image_level_feature = UpSampling2D(size=(h, w), interpolation='bilinear')(image_level_feature)

                aspp_conv_01 = conv_bn_relu(filters=256, kernel_size=(1, 1), dilation_rate=1, name='conv01_r1')(tensor)
                aspp_conv_02 = sep_conv_bn_relu(filters=256, kernel_size=(3, 3), dilation_rate=6, name='atrous01_r6')(
                    tensor)
                aspp_conv_03 = sep_conv_bn_relu(filters=256, kernel_size=(3, 3), dilation_rate=12, name='atrous02_r12')(
                    tensor)
                aspp_conv_04 = sep_conv_bn_relu(filters=256, kernel_size=(3, 3), dilation_rate=18, name='atrous03_r18')(
                    tensor)

                x = Concatenate()([aspp_conv_01, aspp_conv_02, aspp_conv_03, aspp_conv_04, image_level_feature])
                return x

            return f

        def deep_lab_v3_plus(backbone='xception', OS=16, shape=(None, None, 3), num_classes=1):
            """
            This method returns a Keras image segmentation model.

            Parameters
            --------
            shape: shape tuple, in "channels_last" format;
                it should have exactly 3 inputs channels, and width and
                height should be no smaller than 32.
                E.g. `(256, 256, 3)` would be one valid value. Default to `None`.
            backbone: string
                it should have a value from this set: ['xception', 'resnet50',  'resnet101', 'mobilenetv2']
            OS: int
                output stride
                commonly take value 8, 16
            num_classes: int
                in segmentation determine number of label in each pixel (pixel classification)

            Returns
            -------
            A `Tensorflow.keras.Model` instance.
            """

            if backbone == 'vgg16':
                """
                # Not Implemented
                backbone_model = VGG16Net(input_shape=shape, OS=OS)
                high_level_feature_layer = backbone_model.layers[-1].output
                low_level_feature_layer = backbone_model.get_layer('block3_conv3').output
                """
                pass
            elif backbone == 'xception':
                backbone_model = Xception(input_shape=shape)
                if OS == 16:
                    low_level_feature_layer = backbone_model.get_layer('Add_entry_block-A').output
                    high_level_feature_layer = backbone_model.layers[-1].output
            elif backbone == 'mobilenetv2':
                """
                # Not Implemented
                backbone_model = MobileNetV2_modified(input_shape=shape, OS=OS)
                high_level_feature_layer = backbone_model.layers[-1].output
                low_level_feature_layer = backbone_model.get_layer('block_3_expand_relu').output
                """
                pass
            elif backbone == 'resnet101':
                """
                # Not Implemented
                backbone_model = ResNet_101(input_shape=shape, OS=OS)
                backbone_model.load_weights(resnet101_weight_path, by_name=True)
                high_level_feature_layer = backbone_model.layers[-1].output
                low_level_feature_layer = backbone_model.get_layer('conv2_block3_out').output
                """
                pass
            elif backbone == 'resnet50':
                """
                # Not Implemented
                backbone_model = ResNet_50(input_shape=shape, OS=OS)
                high_level_feature_layer = backbone_model.layers[-1].output
                low_level_feature_layer = backbone_model.get_layer('conv2_block3_out').output
                """
                pass
            else:
                raise ValueError(
                    'Implementation for backbone "{}" is not present. Please choose from "vgg16, xception, mobilenetv2,' \
                    ' resnet50 and resnet101"'.format(backbone))

            x = ASPP()(high_level_feature_layer)

            x = conv_bn_relu(filters=256, kernel_size=(1, 1), name='feature_reduce_high_level')(x)

            if OS == 16:
                x = UpSampling2D(size=(4, 4), interpolation='bilinear', name='upsample_4x_first')(x)
            elif OS == 8:
                x = UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample_2x_first')(x)

            low_level_features = conv_bn_relu(filters=48, kernel_size=(1, 1), name='bottleneck_low_level_features')(
                low_level_feature_layer)

            x = Concatenate()([x, low_level_features])
            x = sep_conv_bn_relu(filters=256, kernel_size=(3, 3), name='sep_conv_last')(x)

            x = UpSampling2D(size=(4, 4), interpolation='bilinear', name='upsample_4x_second')(x)

            x = Conv2D(filters=num_classes, kernel_size=(1, 1), padding='same', name='logit')(x)
            x = Activation('sigmoid')(x)

            return Model(inputs=backbone_model.input, outputs=x)

        return deep_lab_v3_plus(backbone=self.backbone, OS=self.OS, shape=self.shape, num_classes=self.num_classes)


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        backbone = 'xception'
        output_stride = 16
        model = DeepLabV3Plus(backbone=backbone,
                              input_shape=(512, 512, 3),
                              OS=output_stride)
        model.get_model().summary()
