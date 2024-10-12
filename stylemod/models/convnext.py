from stylemod.core.cnn import CNNBaseModel
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXt_Tiny(CNNBaseModel):

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
            normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            eval_mode=False,
            retain_graph=False
        )
