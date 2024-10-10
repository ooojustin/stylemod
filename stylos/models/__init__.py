from enum import Enum
from stylos.models.model import Model
from stylos.models.vgg19 import VGG19
from stylos.models.efficientnet_b0 import EFFICIENTNET_B0
from stylos.models.vit_b_16 import VIT_B_16
from stylos.models.resnet50 import RESNET50
from stylos.models.convnext import CONVNEXT_TINY


class Models(Enum):
    VGG19 = VGG19
    EFFICIENTNET_B0 = EFFICIENTNET_B0
    VIT_B_16 = VIT_B_16
    RESNET50 = RESNET50
    CONVNEXT_TINY = CONVNEXT_TINY

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
