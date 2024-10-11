from stylemod.models.base_model import BaseModel
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXt_Tiny(BaseModel):

    def __init__(self):
        super().__init__(
            model_fn=convnext_tiny,
            weights=ConvNeXt_Tiny_Weights.DEFAULT,
            name="ConvNeXt_Tiny",
            content_layer="4",
            style_weights={
                "0": 1.0,
                "1": 0.8,
                "3": 0.6,
                "4": 0.4,
                "5": 0.2
            },
            eval_mode=False,
            retain_graph=False
        )
