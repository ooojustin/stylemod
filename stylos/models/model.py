import torch
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Optional
from torchvision.models._api import Weights


@dataclass
class Model:
    name: str
    model_fn: Callable[..., Any]
    weights: Weights
    content_layer: str
    style_layers: List[str]
    style_weights: Dict[str, float]
    model: Optional[torch.nn.Module] = field(default=None, init=False)
    eval_mode: bool = field(default=False)
    retain_graph: bool = field(default=False)

    def init(self):
        model = self.model_fn(weights=self.weights)

        # NOTE(justin): not all models have a features attribute (ex: ResNet, ViT)
        if isinstance(model, torch.nn.Module) and hasattr(model, 'features'):
            model = model.features

        for param in model.parameters():
            param.requires_grad_(False)

        self.model = model

    def get_model_module(self) -> torch.nn.Module:
        if self.model is None:
            self.init()
        assert self.model is not None
        return self.model

    def set_device(self, device: torch.device) -> torch.nn.Module:
        self.model = self.get_model_module().to(device)
        return self.model

    def eval(self):
        model = self.get_model_module()
        self.model = model.eval()

    def get_features(self, image, layers):
        features = {}
        model = self.get_model_module()

        x = image
        for name, layer in model._modules.items():
            assert layer
            x = layer(x)
            if name in layers:
                features[name] = x

        return features

    def gram_matrix(self, tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        gram /= h * w  # TODO(justin): make gram matrix normalization optional
        return gram
