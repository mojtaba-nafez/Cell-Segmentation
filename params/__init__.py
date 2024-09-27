from .unet import unet_args as args_unet
from .deeplabv3plus_resnet import deeplabv3p_resnet_args
from .deeplabv3plus_xception import deeplabv3plus_xception_args
from .deeplabV3plus_xception_attention import deeplabV3plus_xception_attention_args
from .deeplabV3plus_resnet_attention import deeplabV3plus_resnet_attention_args
from .unet_pretrain import unet_pretrain_args
from .unet_pyramid_cbam import unet_pyramid_cbam_args
from .unet_pretrain_cbam import unet_pretrain_cbam_args

ARGUMENTS = dict(
    unet=args_unet,
    unet_pretrain=unet_pretrain_args,
    unet_bn=args_unet,
    deeplabv3plus_resnet=deeplabv3p_resnet_args,
    deeplabv3plus_xception=deeplabv3plus_xception_args,
    deeplabV3plus_xception_attention=deeplabV3plus_xception_attention_args,
    deeplabV3plus_resnet_attention=deeplabV3plus_resnet_attention_args,
    unet_pyramid_cbam=unet_pyramid_cbam_args,
    unet_pretrain_cbam=unet_pretrain_cbam_args,
)


def get_args(model_name):
    """Get Argument Parser"""
    return ARGUMENTS[model_name]()
