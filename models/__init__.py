# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""
This module load user selected model.

this modules contain following Functions:
    -   load_model
"""

from .unet import UNet as Unet
from .unet_pretrain import UNet as UnetPR
from .unet_batch_normalization import UNet as UnetBN
from .deeplabv3plus_resnet import DeepLabV3Plus as deeplabv3plus_resnet
from .deeplabv3plus_xception import DeepLabV3Plus as deeplabv3plus_xception
from .deeplabv3plus_resnet_attention import DeepLabV3PlusAttention as DeepLabV3PlusResnetAttention
from .deeplabv3plus_xception_attention import DeepLabV3PlusAttention as DeepLabV3PlusXceptionAttention
from .unet_pyramid_cbam import UNetPyramidCBAM
from .unet_pretrain_cbam import UNetPretrainCBAM

MODELS = dict(
    unet=Unet,
    unet_bn=UnetBN,
    unet_pretrain=UnetPR,
    deeplabv3plus_resnet=deeplabv3plus_resnet,
    deeplabv3plus_xception=deeplabv3plus_xception,
    deeplabV3plus_resnet_attention=DeepLabV3PlusResnetAttention,
    deeplabV3plus_xception_attention=DeepLabV3PlusXceptionAttention,
    unet_pyramid_cbam=UNetPyramidCBAM,
    unet_pretrain_cbam=UNetPretrainCBAM
)


def load_model(model_name, **kwargs):
    """ Get model
       Parameters
       ----------
       model_name  string, user selected model name

       Returns
       -------
       model     tensorflow model instance
       """
    return MODELS[model_name](**kwargs).get_model()
