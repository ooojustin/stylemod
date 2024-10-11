from stylemod.models.factory import ModelFactory
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
    ModelFactory.register("VGG19", VGG19)
    ModelFactory.register("EFFICIENTNET_B0", EfficientNetB0)
    ModelFactory.register("EFFICIENTNET_V2", EfficientNetV2)
    ModelFactory.register("VIT_B_16", ViT_B_16)
    ModelFactory.register("RESNET50", ResNet50)
    ModelFactory.register("CONVNEXT_TINY", ConvNeXt_Tiny)
    ModelFactory.register("SWIN_T", Swin_T)
    ModelFactory.register("DENSENET121", DenseNet121)
    ModelFactory.register("REGNET_Y_16GF", RegNet_Y_16GF)


register_default_models()
