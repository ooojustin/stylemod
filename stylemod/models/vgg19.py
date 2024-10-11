from stylemod.core.base_model import BaseModel
from torchvision.models import vgg19, VGG19_Weights


class VGG19(BaseModel):

    def __init__(self):
        super().__init__(
            model_fn=vgg19,
            weights=VGG19_Weights.DEFAULT,
            name="VGG19",
            content_layer="21",
            style_weights={
                "0": 1.0,
                "5": 0.8,
                "10": 0.5,
                "19": 0.3,
                "28": 0.1
            },
            eval_mode=False,
            retain_graph=False
        )
