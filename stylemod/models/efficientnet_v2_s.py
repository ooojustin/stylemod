from stylemod.core.factory import ModelFactory
from stylemod.core.base_model import BaseModel
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class EfficientNetV2(BaseModel):

    def __init__(self):
        super().__init__(
            model_fn=efficientnet_v2_s,
            weights=EfficientNet_V2_S_Weights.DEFAULT,
            name="EfficientNetV2",
            content_layer="4",
            style_weights={
                "0": 1.0,
                "1": 0.8,
                "2": 0.6,
                "4": 0.4,
                "6": 0.2
            },
            eval_mode=False,
            retain_graph=False
        )


ModelFactory.register("EfficientNetV2", EfficientNetV2)
