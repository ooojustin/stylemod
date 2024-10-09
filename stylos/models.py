import torch
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Optional
from enum import Enum
from torch import nn
from torchvision.models._api import Weights
from torchvision.models import (
    vgg19, VGG19_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)


@dataclass
class Model:
    name: str
    model_fn: Callable[..., Any]
    weights: Weights
    content_layer: str
    style_layers: List[str]
    style_weights: Dict[str, float]
    model: Optional[nn.Module] = field(default=None, init=False)

    def init(self):
        model = self.model_fn(weights=self.weights)

        # NOTE(justin): ResNet/ViT don"t have a features attribute
        model = model.features

        for param in model.parameters():
            param.requires_grad_(False)

        self.model = model

    def get_model_module(self, device: Optional[torch.device] = None) -> nn.Module:
        if self.model is None:
            self.init()
        assert self.model is not None
        return self.model.to(device) if device \
            else self.model

    @staticmethod
    def load(model_name: str) -> "Model":
        try:
            model_name = model_name.upper()
            model = ModelEnum[model_name].value
        except KeyError:
            raise ValueError(
                f"Model {model_name} not supported. Choose from: {[e.name for e in ModelEnum]}"
            )
        model.init()
        return model


class ModelEnum(Enum):
    VGG19 = Model(
        name="VGG19",
        model_fn=vgg19,
        weights=VGG19_Weights.DEFAULT,  # type: ignore[assignment]
        content_layer="21",
        style_layers=["0", "5", "10", "19", "28"],
        style_weights={
            # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            "0": 1.0,
            # Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            "5": 0.8,
            # Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            "10": 0.5,
            # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            "19": 0.3,
            # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            "28": 0.1
        }
    )
    EFFICIENTNET_B0 = Model(
        name="EFFICIENTNET_B0",
        model_fn=efficientnet_b0,
        weights=EfficientNet_B0_Weights.DEFAULT,  # type: ignore[assignment]
        content_layer="6",
        style_layers=["0", "2", "4", "6"],
        style_weights={
            # Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            "0": 1.0,
            # MBConv(Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False), Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False))
            "2": 0.8,
            # MBConv(Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False), Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False))
            "4": 0.5,
            # MBConv(Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False), Conv2d(240, 240, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=240, bias=False))
            "6": 0.3,
            # Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
            "8": 0.1
        }

    )
