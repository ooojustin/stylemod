from stylemod.core.cnn import CNNBaseModel
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class EfficientNetV2(CNNBaseModel):

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
            normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            eval_mode=False,
            retain_graph=False
        )
