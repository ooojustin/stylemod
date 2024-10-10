from enum import Enum
from stylos.models.base_model import BaseModel
from stylos.models.vgg19 import VGG19
from stylos.models.efficientnet_b0 import EfficientNetB0
from stylos.models.efficientnet_v2_s import EfficientNetV2
from stylos.models.vit_b_16 import ViT_B_16
from stylos.models.resnet50 import ResNet50
from stylos.models.convnext import ConvNeXt_Tiny
from stylos.models.swin_t import Swin_T
from stylos.models.densenet121 import DenseNet121
from stylos.models.regnet_y_16gf import RegNet_Y_16GF


class Models(Enum):
    VGG19 = VGG19()
    EFFICIENTNET_B0 = EfficientNetB0()
    EFFICIENTNET_V2 = EfficientNetV2()
    VIT_B_16 = ViT_B_16()
    RESNET50 = ResNet50()
    CONVNEXT_TINY = ConvNeXt_Tiny()
    SWIN_T = Swin_T()
    DENSENET121 = DenseNet121()
    REGNET_Y_16GF = RegNet_Y_16GF()

    @staticmethod
    def load(model_name: str) -> BaseModel:
        try:
            model_name = model_name.upper()
            model = Models[model_name].value
        except KeyError:
            raise ValueError(
                f"Model {model_name} not supported. Choose from: {[e.name for e in Models]}"
            )
        model.initialize_module()
        return model
