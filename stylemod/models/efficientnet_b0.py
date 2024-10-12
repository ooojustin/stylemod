from stylemod.core.cnn import CNNBaseModel
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0(CNNBaseModel):

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
            normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            eval_mode=False,
            retain_graph=False
        )
