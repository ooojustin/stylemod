import torch
from stylemod.core.base import DEFAULTS
from stylemod.core.cnn import CNNBaseModel
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict


class ResNet50(CNNBaseModel):

    def __init__(
        self,
        model_fn=resnet50,
        weights=ResNet50_Weights.DEFAULT,
        content_layer="layer4",
        style_weights={
            "conv1": 1.0,
            "layer1": 0.8,
            "layer2": 0.6,
            "layer3": 0.4,
            "layer4": 0.2
        },
        content_weight: float = DEFAULTS["content_weight"],
        style_weight: float = DEFAULTS["style_weight"],
        learning_rate: float = DEFAULTS["learning_rate"],
        normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        eval_mode=False,
        retain_graph=False
    ):
        super().__init__(
            name="ResNet50",
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

    def get_features(self, image: torch.Tensor, layers: list) -> dict:
        features: Dict[str, torch.Tensor] = {}
        model = self.get_model_module()
        x = image
        for name, layer in model._modules.items():
            assert layer
            x = layer(x)
            if name in layers:
                features[name] = x
            # stop before fc layer
            if name == "avgpool":
                break
        return features
