from stylemod.core.base import DEFAULTS
from stylemod.core.cnn import CNNBaseModel
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXt_Tiny(CNNBaseModel):

    def __init__(
        self,
        model_fn=convnext_tiny,
        weights=ConvNeXt_Tiny_Weights.DEFAULT,
        content_layer="4",
        style_weights={
            "0": 1.0,
            "1": 0.8,
            "3": 0.6,
            "4": 0.4,
            "5": 0.2
        },
        content_weight: float = DEFAULTS["content_weight"],
        style_weight: float = DEFAULTS["style_weight"],
        learning_rate: float = DEFAULTS["learning_rate"],
        normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        eval_mode=False,
        retain_graph=False
    ):
        super().__init__(
            name="ConvNeXt_Tiny",
            model_fn=model_fn,
            weights=weights,
            content_layer=content_layer,
            style_weights=style_weights,
            content_weight=content_weight,
            style_weight=style_weight,
            learning_rate=learning_rate,
            normalization=normalization,
            eval_mode=eval_mode,
            retain_graph=retain_graph
        )
