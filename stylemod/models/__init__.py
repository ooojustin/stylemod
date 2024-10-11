from enum import Enum
from stylemod.models.vgg19 import VGG19
from stylemod.models.efficientnet_b0 import EfficientNetB0
from stylemod.models.efficientnet_v2_s import EfficientNetV2
from stylemod.models.vit_b_16 import ViT_B_16
from stylemod.models.resnet50 import ResNet50
from stylemod.models.convnext import ConvNeXt_Tiny
from stylemod.models.swin_t import Swin_T
from stylemod.models.densenet121 import DenseNet121
from stylemod.models.regnet_y_16gf import RegNet_Y_16GF


class Model(Enum):
    VGG19 = VGG19
    EFFICIENTNET_B0 = EfficientNetB0
    EFFICIENTNET_V2 = EfficientNetV2
    VIT_B_16 = ViT_B_16
    RESNET50 = ResNet50
    CONVNEXT_TINY = ConvNeXt_Tiny
    SWIN_T = Swin_T
    DENSENET121 = DenseNet121
    REGNET_Y_16GF = RegNet_Y_16GF
