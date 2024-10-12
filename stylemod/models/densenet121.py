from stylemod.core.cnn import CNNBaseModel
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNet121(CNNBaseModel):

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
            normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            eval_mode=False,
            retain_graph=False
        )
