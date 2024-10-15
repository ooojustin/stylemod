from stylemod.core.cnn import CNNBaseModel
from torchvision.models import vgg19, VGG19_Weights


class VGG19(CNNBaseModel):

    def __init__(
        self,
        model_fn=vgg19,
        weights=VGG19_Weights.DEFAULT,
        content_layer="21",
        style_weights={
            "0": 1.0,
            "5": 0.8,
            "10": 0.5,
            "19": 0.3,
            "28": 0.1
        },
        normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        eval_mode=False,
        retain_graph=False
    ):
        super().__init__(
            name="VGG19",
            model_fn=model_fn,
            weights=weights,
            content_layer=content_layer,
            style_weights=style_weights,
            normalization=normalization,
            eval_mode=eval_mode,
            retain_graph=retain_graph
        )
