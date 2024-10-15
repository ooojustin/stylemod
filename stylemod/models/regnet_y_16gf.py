import torch
import torchvision
from stylemod.core.cnn import CNNBaseModel
from typing import Dict, List
from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights


class RegNet_Y_16GF(CNNBaseModel):

    def __init__(
        self,
        model_fn=regnet_y_16gf,
        weights=RegNet_Y_16GF_Weights.DEFAULT,
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
    ):
        super().__init__(
            name="RegNetY16GF",
            model_fn=model_fn,
            weights=weights,
            content_layer=content_layer,
            style_weights=style_weights,
            normalization=normalization,
            eval_mode=eval_mode,
            retain_graph=retain_graph
        )

    def get_features(self, image: torch.Tensor, layers: List[str]) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        model = self.get_model_module()
        x = image
        for name, layer in model.named_children():
            x = layer(x)
            if name == "trunk_output":
                for trunk_name, trunk_layer in layer.named_children():
                    # this is primarily to fix the transition from block4 (3024 channels) to block1 (32 channels).
                    # a 1x1 convolution is used to reduce the number of channels, but the logic is applied dynamically
                    if isinstance(trunk_layer, torchvision.models.regnet.AnyStage):
                        x = self.__fix_conv2d_channels(trunk_layer, x)
                    x = trunk_layer(x)
                    if trunk_name in layers:
                        features[trunk_name] = x
            if name in layers:
                features[name] = x
            # stop before fc layer
            if name == "avgpool":
                break
        return features

    def __fix_conv2d_channels(self, layer: torchvision.models.regnet.AnyStage, tensor: torch.Tensor) -> torch.Tensor:
        device = tensor.device
        for _, block in layer.named_children():
            for c_layer, _ in [
                layer for _, layer in block.named_children()
                if isinstance(layer, torchvision.ops.misc.Conv2dNormActivation)
            ]:
                if isinstance(c_layer, torch.nn.Conv2d) and tensor.shape[1] != c_layer.in_channels:
                    adjust = torch.nn.Conv2d(
                        tensor.shape[1], c_layer.in_channels, kernel_size=1).to(device)
                    return adjust(tensor)
        return tensor
