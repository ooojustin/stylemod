from stylemod.core.factory import ModelFactory
from stylemod.core.base_model import BaseModel
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNet121(BaseModel):

    def __init__(self):
        super().__init__(
            model_fn=densenet121,
            weights=DenseNet121_Weights.DEFAULT,
            name="DenseNet121",
            content_layer="denseblock4",
            style_weights={
                "conv0": 1.0,
                "denseblock1": 0.8,
                "denseblock2": 0.6,
                "denseblock3": 0.4,
                "denseblock4": 0.2
            },
            eval_mode=False,
            retain_graph=False
        )


ModelFactory.register("DenseNet121", DenseNet121)
