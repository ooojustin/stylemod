from stylemod.core.factory import ModelFactory
from stylemod.core.base_model import BaseModel
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0(BaseModel):

    def __init__(self):
        super().__init__(
            model_fn=efficientnet_b0,
            weights=EfficientNet_B0_Weights.DEFAULT,
            name="EfficientNetB0",
            content_layer="6",
            style_weights={
                "0": 1.0,
                "2": 0.8,
                "4": 0.5,
                "6": 0.3
            },
            eval_mode=False,
            retain_graph=False
        )


ModelFactory.register("EfficientNetB0", EfficientNetB0)
