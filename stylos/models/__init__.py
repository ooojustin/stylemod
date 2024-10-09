from enum import Enum
from stylos.models.model import Model
from stylos.models.vgg19 import VGG19
from stylos.models.efficientnet_b0 import EFFICIENTNET_B0
from stylos.models.vit_b_16 import VIT_B_16


class Models(Enum):
    VGG19 = VGG19
    EFFICIENTNET_B0 = EFFICIENTNET_B0
    VIT_B_16 = VIT_B_16

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
