# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""Xception model implemented from scratch using tensorflow.keras

Xception is a new architecture and has many use cases in image classification tasks.

It supports various image sizes starting from 1*1

This model just used in Encoder part of deeplabv3plus, so we have not any execution result on it.

Reference:
  - [code](https://github.com/puruBHU/DeepLabv3plus-keras/)
"""
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, SeparableConv2D, Activation, BatchNormalization, Add, Input
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

if K.image_data_format() == 'channels_last':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3


def _bn_relu(input_):
    """
    This method returns a Keras convolution block.
    you can use it to build a model.

    Parameters
    -------
    input_: KerasTensor, input layer

    Returns
    -------
    A `KerasTensor` instance.
    """
    norm = BatchNormalization()(input_)
    return Activation('relu')(norm)


def conv_bn_relu(**params):
    """
    This method returns a Keras convolution block.
    you can use it to build a model.

    Parameters
    -------
    params:
        filters:        int, number of conv filter
        kernel_size:    (int, int), size of conv kernel
        strides:        (int, int), conv stride
        padding:        (int, int), padding size of conv
        dilation_rate:  int, atrous convolution rate (https://stackoverflow.com/questions/63073760/using-dilated-convolution-in-keras)
        kernel_initializer: (int, int)kernel weights initializer
        kernel_regularizer: string or function, nkernel weight regularizer
        activation:     string or function, activation function
        name:           string, layer name

    Returns
    -------
    A `KerasTensor` instance.
    """
    filters = params['filters']
    kernel_size = params['kernel_size']
    strides = params.setdefault('strides', (1, 1))
    padding = params.setdefault('padding', 'same')
    dilation_rate = params.setdefault('dilation_rate', 1)
    kernel_initializer = params.setdefault('kernel_initializer', he_normal())
    kernel_regularizer = params.setdefault('kernel_regularizer', l2(1e-3))
    activation = params.setdefault('activation', 'relu')
    name = params.setdefault('name', None)

    if not name == None:
        conv_name = 'conv_{}'.format(name)
        bn_name = 'BN_{}'.format(name)
        act_name = 'Act_{}_{}'.format(name, activation)

    def f(input_):
        """
        This method returns a Keras convolution block.
        you can use it to build a model.

         Parameters
        -------
        input_: KerasTensor, input layer

        Returns
        -------
        A `KerasTensor` instance.
        """
        conv = Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      name=conv_name)(input_)

        batch_norm = BatchNormalization(name=bn_name)(conv)

        return Activation(activation, name=act_name)(batch_norm)

    return f


def sep_conv_bn_relu(**params):
    """
    This method returns a Keras convolution block.
    you can use it to build a model.

    Parameters
    -------
    params:
        filters:        int, number of conv filter
        kernel_size:    (int, int), size of conv kernel
        strides:        (int, int), conv stride
        padding:        (int, int), padding size of conv
        dilation_rate:  int, atrous convolution rate (https://stackoverflow.com/questions/63073760/using-dilated-convolution-in-keras)
        depthwise_initializer: string or function, kernel weights initializer
        pointwise_initializer: string or function, kernel weights initializer
        depthwise_regularizer: string or function, kernel weight regularizer
        pointwise_regularizer: string or function, kernel weight regularizer
        activation:     string, activation function
        name:           KerasTensor, layer name

    Returns
    -------
    A `KerasTensor` instance.
    """
    filters = params['filters']
    kernel_size = params.setdefault('kernel_size', (3, 3))
    strides = params.setdefault('strides', (1, 1))
    padding = params.setdefault('padding', 'same')
    dilation_rate = params.setdefault('dilation_rate', 1)
    depthwise_initializer = params.setdefault('depthwise_initializer', he_normal())
    pointwise_initializer = params.setdefault('pointwise_initializer', he_normal())

    depthwise_regularizer = params.setdefault('depthwise_regularizer', l2(1e-3))
    pointwise_regularizer = params.setdefault('pointwise_regularizer', l2(1e-3))
    activation = params.setdefault('activation', 'relu')
    name = params.setdefault('name', None)

    if not name == None:
        conv_name = 'conv_{}'.format(name)
        bn_name = 'BN_{}'.format(name)
        act_name = 'Act_{}_{}'.format(name, activation)

    def f(input_):
        """
         This method returns a Keras convolution block.
         you can use it to build a model.

         Parameters
         -------
         input_: KerasTensor, input layer

         Returns
         -------
         A `KerasTensor` instance.
         """
        conv = SeparableConv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               dilation_rate=dilation_rate,
                               depthwise_initializer=depthwise_initializer,
                               pointwise_initializer=pointwise_initializer,
                               depthwise_regularizer=depthwise_regularizer,
                               pointwise_regularizer=pointwise_regularizer,
                               name=conv_name
                               )(input_)

        batch_norm = BatchNormalization(name=bn_name)(conv)

        return Activation(activation, name=act_name)(batch_norm)

    return f


def _shortcut(input_, residual, name=None):
    """
    Add a shorcut between input and residula block and merges then with Add

    This method returns a Keras convolution block.
    you can use it to build a model.

    Parameters
    -------
    input_:      , input shape
    residual:   , residual_shape
    name:       string, layer name

    Returns
    -------
    A `KerasTensor` instance.
    """
    input_shape = K.int_shape(input_)
    residual_shape = K.int_shape(residual)

    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))

    is_equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shorcut = input_

    if stride_width > 1 or stride_height > 1 or not is_equal_channels:
        shorcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                         kernel_size=(1, 1),
                         strides=(stride_width, stride_height),
                         padding='same',
                         kernel_initializer=he_normal(),
                         kernel_regularizer=l2(1e-4),
                         name='shortcut_{}'.format(name))(input_)

    return Add(name='Add_{}'.format(name))([shorcut, residual])


def basic_block(filters, init_strides=(1, 1), name=None):
    """
    This method returns a Keras convolution block.
    you can use it to build a model.

    Parameters
    -------
    filters:        int, number of conv filter
    init_strides:   (int, int), conv stride
    name:           string, layer name

    Returns
    -------
    A `KerasTensor` instance.
    """

    def f(input_):
        """
         This method returns a Keras convolution block.
         you can use it to build a model.

         Parameters
         -------
         input_: KerasTensor, input layer

         Returns
         -------
         A `KerasTensor` instance.
         """
        residual = sep_conv_bn_relu(filters=filters, strides=init_strides, name=name)(input_)
        return residual

    return f


def _residual_block(block_function, filters, repetition=3, increase_stride=False, name=None):
    """
    This method returns a Keras convolution block.
    you can use it to build a model.

    Parameters
    -------
    block_function: function, function of conv block
    filters:        int, number of conv filter
    repetition:
    increase_stride:
    name:           string, layer name

    Returns
    -------
    A `KerasTensor` instance.
    """

    def f(input_):
        """
         This method returns a Keras convolution block.
         you can use it to build a model.

         Parameters
         -------
         input_: KerasTensor, input layer

         Returns
         -------
         A `KerasTensor` instance.
         """
        residual = input_
        for i in range(repetition):
            strides = (1, 1)
            if i == (repetition - 1) and increase_stride:
                strides = (2, 2)

            residual = block_function(filters=filters, init_strides=strides, name='{0}_{1:02d}'.format(name, i + 1))(
                residual)
        return _shortcut(input_, residual, name=name)

    return f


def entry_residual_block(block_function):
    """
    This method returns a Keras convolution block.
    you can use it to build a model.

    Parameters
    -------
    block_function: function, function of conv block

    Returns
    -------
    A `KerasTensor` instance.
    """

    def f(input_):
        """
         This method returns a Keras convolution block.
         you can use it to build a model.

         Parameters
         -------
         input_: KerasTensor, input layer

         Returns
         -------
         A `KerasTensor` instance.
         """
        x = _residual_block(block_function, filters=128, repetition=3, increase_stride=True, name='entry_block-A')(
            input_)
        x = _residual_block(block_function, filters=256, increase_stride=True, name='entry_block-B')(x)
        x = _residual_block(block_function, filters=728, increase_stride=True, name='entry_block-C')(x)
        return x

    return f


def middle_residual_block(block_function, reptition=16):
    """
    This method returns a Keras convolution block.
    you can use it to build a model.

    Parameters
    -------
    block_function
    reptition

    Returns
    -------
    A `KerasTensor` instance.
    """

    def f(input_):
        """
         This method returns a Keras convolution block.
         you can use it to build a model.

         Parameters
         -------
         input_: KerasTensor, input layer

         Returns
         -------
         A `KerasTensor` instance.
         """
        for i in range(reptition):
            input_ = _residual_block(block_function, filters=728, increase_stride=False,
                                     name='middle_block_{:02d}'.format(i + 1))(input_)
        return input_

    return f


def exit_block(block_function, final_block_stride=(1, 1), rate=2):
    """
    This method returns a Keras convolution block.
    you can use it to build a model.

    Parameters
    -------
    block_function      KerasTensor,
    final_block_stride  KerasTensor,
    rate                int, not used

    Returns
    -------
    A `KerasTensor` instance.
    """

    def f(input_):
        """
         This method returns a Keras convolution block.
         you can use it to build a model.

         Parameters
         -------
         input_: KerasTensor, input layer

         Returns
         -------
         A `KerasTensor` instance.
         """
        x = block_function(filters=728, name='exit_res_01')(input_)
        x = block_function(filters=1024, name='exit_res_02')(x)
        x = block_function(filters=1024, init_strides=final_block_stride, name='exit_res_03')(x)
        x = _shortcut(input_, x, name='exit_block')

        if final_block_stride[0] == 1:
            x = sep_conv_bn_relu(filters=1536, dilation_rate=2, name='exit_A')(x)
            x = sep_conv_bn_relu(filters=1536, dilation_rate=2, name='exit_B')(x)
            x = sep_conv_bn_relu(filters=2048, dilation_rate=2, name='exit_c')(x)
        else:
            x = block_function(filters=1536, name='exit_A')(x)
            x = block_function(filters=1536, name='exit_B')(x)
            x = block_function(filters=2048, name='exit_C')(x)
        return x

    return f


def Xception(input_shape=(None, None, 3)):
    """
    This method returns a Keras image segmentation model.

    input_shape: (int, int, 3), input image shapes

    Returns
    -------
    A `Tensorflow.keras.Model` instance.
    """
    input_ = Input(shape=input_shape, name='input_layer')
    x = conv_bn_relu(filters=32, kernel_size=(3, 3), strides=(2, 2), name='input_A')(input_)
    x = conv_bn_relu(filters=64, kernel_size=(3, 3), name='input_B')(x)

    # The entry block
    x = entry_residual_block(block_function=basic_block)(x)
    x = middle_residual_block(block_function=basic_block, reptition=16)(x)
    x = exit_block(block_function=basic_block, final_block_stride=(1, 1), rate=2)(x)

    return Model(inputs=input_, outputs=x)


if __name__ == '__main__':
    model = Xception(input_shape=(512, 512, 3))
    plot_model(model=model, to_file='Xception_deepLabV3.png', show_shapes=True, show_layer_names=True)
    model.summary()
