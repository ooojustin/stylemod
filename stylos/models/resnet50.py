from dataclasses import dataclass
from stylos.models.model import Model
from torchvision.models import resnet50, ResNet50_Weights


@dataclass
class ResNetModel(Model):
    def get_features(self, image, layers):
        """Extract features from specified layers for ResNet."""
        features = {}
        x = image
        model = self.get_model_module()

        for name, layer in model._modules.items():
            assert layer
            x = layer(x)
            if name in layers:
                features[name] = x

            # NOTE(justin): stop forward pass before reaching the fully connected layer (avgpool is right before fc)
            # global average pooling reduces the spatial dimensions (height, width) to 1x1, flattening the feature map into a 1D tensor for each channel
            if name == 'avgpool':
                break

        return features


RESNET50 = ResNetModel(
    name="RESNET50",
    model_fn=resnet50,
    weights=ResNet50_Weights.DEFAULT,  # type: ignore[assignment]
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
    }
)
