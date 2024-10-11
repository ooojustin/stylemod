import torch
import torchvision
from stylemod.core.base_model import BaseModel
from typing import Dict, List
from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights


class RegNet_Y_16GF(BaseModel):

    def __init__(self):
        super().__init__(
            model_fn=regnet_y_16gf,
            weights=RegNet_Y_16GF_Weights.DEFAULT,
            name="RegNetY16GF",
            content_layer="trunk_output",
            style_weights={
                "stem": 1.0,
                "block1": 0.8,
                "block2": 0.6,
                "block3": 0.4,
                "block4": 0.2
            },
            normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            eval_mode=True,
            retain_graph=True
        )

    def get_features(self, image: torch.Tensor, layers: List[str]) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        model = self.get_model_module()

        x = image
        for name, layer in model.named_children():
            x = layer(x)

            if name == "trunk_output":
                for trunk_name, trunk_layer in layer.named_children():

                    # NOTE(justin):
                    # this is primarily to fix the transition from block4 (3024 channels) to block1 (32 channels).
                    # a 1x1 convolution is used to reduce the number of channels, but the logic is applied dynamically.
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
        device = tensor.device

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
                        tensor.shape[1], conv_layer.in_channels, kernel_size=1).to(device)
                    return adjust_channels(tensor)

        return tensor
