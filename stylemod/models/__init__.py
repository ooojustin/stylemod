from enum import Enum
from stylemod.models.factory import ModelFactory
from stylemod.models.base_model import BaseModel
from stylemod.models.vgg19 import VGG19
from stylemod.models.efficientnet_b0 import EfficientNetB0
from stylemod.models.efficientnet_v2_s import EfficientNetV2
from stylemod.models.vit_b_16 import ViT_B_16
from stylemod.models.resnet50 import ResNet50
from stylemod.models.convnext import ConvNeXt_Tiny
from stylemod.models.swin_t import Swin_T
from stylemod.models.densenet121 import DenseNet121
from stylemod.models.regnet_y_16gf import RegNet_Y_16GF


def register_default_models():
    ModelFactory.register_model("VGG19", VGG19)
    ModelFactory.register_model("EFFICIENTNET_B0", EfficientNetB0)
    ModelFactory.register_model("EFFICIENTNET_V2", EfficientNetV2)
    ModelFactory.register_model("VIT_B_16", ViT_B_16)
    ModelFactory.register_model("RESNET50", ResNet50)
    ModelFactory.register_model("CONVNEXT_TINY", ConvNeXt_Tiny)
    ModelFactory.register_model("SWIN_T", Swin_T)
    ModelFactory.register_model("DENSENET121", DenseNet121)
    ModelFactory.register_model("REGNET_Y_16GF", RegNet_Y_16GF)


register_default_models()
