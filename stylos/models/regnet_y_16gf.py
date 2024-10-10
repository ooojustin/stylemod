import torch
import torchvision
from typing import Dict, List
from dataclasses import dataclass
from stylos.models.model import Model
from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights


@dataclass
class RegNetModel(Model):

    def get_features(self, image: torch.Tensor, layers: List[str]) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        model = self.get_model_module()

        x = image
        for name, layer in model.named_children():
            x = layer(x)

            if name == "trunk_output":
                for trunk_name, trunk_layer in layer.named_children():

                    # NOTE(justin):
                    # we need to adjust the channels in block1 because the input tensor has 3024 channels, but block1 expects 32.
                    # this mismatch occurs due to the transition from block4 (3024 channels) to block1 (32 channels).
                    # a 1x1 convolution is used to reduce the channels from 3024 to 32, but the logic is applied dynamically.
                    if isinstance(trunk_layer, torchvision.models.regnet.AnyStage):
                        x = self.__fix_conv2d_channels(trunk_layer, x)

                    x = trunk_layer(x)
                    if trunk_name in layers:
                        features[trunk_name] = x

            if name in layers:
                features[name] = x

            # NOTE(justin): stop forward pass before reaching the fully connected layer (avgpool is right before fc)
            if name == "avgpool":
                break

        return features

    def __fix_conv2d_channels(self, layer: torchvision.models.regnet.AnyStage, tensor: torch.Tensor) -> torch.Tensor:
        for _, block in layer.named_children():
            conv2d_norm_layers = [
                layer for _, layer in block.named_children()
                if isinstance(layer, torchvision.ops.misc.Conv2dNormActivation)
            ]

            for norm_layer in conv2d_norm_layers:
                conv_layer = norm_layer[0]
                if not isinstance(conv_layer, torch.nn.Conv2d):
                    continue

                if tensor.shape[1] != conv_layer.in_channels:
                    adjust_channels = torch.nn.Conv2d(
                        tensor.shape[1], conv_layer.in_channels, kernel_size=1)
                    return adjust_channels(tensor)

        return tensor


REGNET_Y_16GF = RegNetModel(
    name="REGNET_Y_16GF",
    model_fn=regnet_y_16gf,
    weights=RegNet_Y_16GF_Weights.DEFAULT,  # type: ignore[assignment]
    content_layer="trunk_output",
    style_layers=[
        "stem",
        "block1",
        "block2",
        "block3",
        "block4"
    ],
    style_weights={
        "stem": 1.0,
        "block1": 0.8,
        "block2": 0.6,
        "block3": 0.4,
        "block4": 0.2
    },
    eval_mode=True,
    retain_graph=True
)
