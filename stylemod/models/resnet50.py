import torch
from stylemod.models.base_model import BaseModel
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50(BaseModel):

    def __init__(self):
        super().__init__(
            model_fn=resnet50,
            weights=ResNet50_Weights.DEFAULT,
            name="ResNet50",
            content_layer="layer4",
            style_layers=[
                "conv1",
                "layer1",
                "layer2",
                "layer3",
                "layer4"
            ],
            style_weights={
                "conv1": 1.0,
                "layer1": 0.8,
                "layer2": 0.6,
                "layer3": 0.4,
                "layer4": 0.2
            },
            eval_mode=False,
            retain_graph=False
        )

    def get_features(self, image: torch.Tensor, layers: list) -> dict:
        features = {}
        x = image
        model = self.get_model_module()

        for name, layer in model._modules.items():
            assert layer
            x = layer(x)
            if name in layers:
                features[name] = x

            # NOTE(justin): stop forward pass before reaching the fully connected layer (avgpool is right before fc)
            if name == 'avgpool':
                break

        return features
