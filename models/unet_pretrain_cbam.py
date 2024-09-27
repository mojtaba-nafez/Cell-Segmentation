from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.applications import ResNet50
from utils.attention import *


class UNetPretrainCBAM:
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
        self.resnet_trainable = kwargs.get("resnet_trainable", None)

    def get_model(self) -> Model:
        """
        This method returns a Keras image segmentation model.

        Returns
        -------
        A `Tensorflow.keras.Model` instance.
        """

        def conv_block(input, num_filters):
            x = Conv2D(num_filters, 3, padding="same")(input)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(num_filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            return x

        def decoder_block(input, skip_features, num_filters):
            x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
            x = Concatenate()([x, skip_features])
            x = conv_block(x, num_filters)
            return x

        def build_resnet50_unet(input_shape):
            """ Input """
            inputs = Input(input_shape)

            """ Pre-trained ResNet50 Model """
            resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
            resnet50.trainable=self.resnet_trainable
            """ Encoder """
            s1 = resnet50.get_layer("input_1").output  ## (512 x 512)
            s2 = resnet50.get_layer("conv1_relu").output  ## (256 x 256)
            s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
            s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)

            """ Bridge """
            b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

            s1 = attach_attention_module(s1, 'cbam_block')
            s2 = attach_attention_module(s2, 'cbam_block')
            s3 = attach_attention_module(s3, 'cbam_block')
            s4 = attach_attention_module(s4, 'cbam_block')

            """ Decoder """
            attn1 = AttnGatingBlock(s4, b1, 64)
            d1 = decoder_block(b1, attn1, 512)  ## (64 x 64)

            attn2 = AttnGatingBlock(s3, d1, 64)
            d2 = decoder_block(d1, attn2, 256)  ## (128 x 128)

            attn3 = AttnGatingBlock(s2, d2, 64)
            d3 = decoder_block(d2, attn3, 128)  ## (256 x 256)

            attn4 = AttnGatingBlock(s1, d3, 64)
            d4 = decoder_block(d3, attn4, 64)  ## (512 x 512)
   
            # d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
            # d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
            # d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
            # d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

            """ Output """
            outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

            model = Model(inputs, outputs, name="ResNet50_U-Net")
            return model

        model = build_resnet50_unet(self.input_shape)
        return model
