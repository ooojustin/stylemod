from enum import Enum
from stylos.models.model import Model
from stylos.models.vgg19 import VGG19
from stylos.models.efficientnet_b0 import EFFICIENTNET_B0
from stylos.models.vit_b_16 import VIT_B_16
from stylos.models.resnet50 import RESNET50
from stylos.models.convnext import CONVNEXT_TINY
from stylos.models.swin_t import SWIN_T
from stylos.models.densenet121 import DENSENET121
from stylos.models.efficientnet_v2_s import EFFICIENTNET_V2


class Models(Enum):
    VGG19 = VGG19
    EFFICIENTNET_B0 = EFFICIENTNET_B0
    VIT_B_16 = VIT_B_16
    RESNET50 = RESNET50
    CONVNEXT_TINY = CONVNEXT_TINY
    SWIN_T = SWIN_T
    DENSENET121 = DENSENET121
    EFFICIENTNET_V2 = EFFICIENTNET_V2

    @staticmethod
    def load(model_name: str) -> Model:
        try:
            model_name = model_name.upper()
            model = Models[model_name].value
        except KeyError:
            raise ValueError(
                f"Model {model_name} not supported. Choose from: {[e.name for e in Models]}"
            )
        model.init()
        return model
