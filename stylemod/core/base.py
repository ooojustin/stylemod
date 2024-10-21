import torch
import graphviz
import warnings
import torchvision.transforms as transforms
from stylemod.core.abstract import AbstractBaseModel, NormalizationType
from typing import Callable, Dict, List, Optional, Any


# TODO(justin): tweak default hyperparameters of each model
DEFAULTS: Dict[str, Any] = {
    "content_weight": 1e4,
    "style_weight": 1e2,
    "learning_rate": 0.003
}


class BaseModel(AbstractBaseModel):
    """
    Provides common functionality like initialization, normalization, feature extraction, and loss calculation.
    Subclasses extend it to focus on model-specific logic.
    """

    def __init__(
        self,
        model_fn: Callable[..., torch.nn.Module],
        weights=None,
        name: str = "",
        content_layer: str = "",
        style_weights: Dict[str, float] = {},  # per layer
        content_weight: float = DEFAULTS["content_weight"],
        style_weight: float = DEFAULTS["style_weight"],
        learning_rate: float = DEFAULTS["learning_rate"],
        normalization: Optional[NormalizationType] = None,
        eval_mode: bool = False,
        retain_graph: bool = False
    ):
        assert callable(model_fn), "'model_fn' must be callable"
        self.name = name
        self.model_fn = model_fn
        self.weights = weights
        self.content_layer = content_layer
        self.style_layers = list(style_weights.keys())
        self.style_weights = style_weights
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.learning_rate = learning_rate
        self.normalization = normalization
        self.eval_mode = eval_mode
        self.retain_graph = retain_graph
        self.model = None

    def initialize_module(self) -> None:
        model = self.model_fn(weights=self.weights)
        if hasattr(model, 'features'):
            model = model.features
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model

    def get_model_module(self) -> torch.nn.Module:
        if self.model is None:
            self.initialize_module()
        assert self.model is not None, "Model initialization failed."
        return self.model

    def eval(self) -> torch.nn.Module:
        model = self.get_model_module()
        self.model = model.eval()
        return self.model

    def set_device(self, device: torch.device) -> torch.nn.Module:
        self.model = self.get_model_module().to(device)
        return self.model

    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.normalization:
            warnings.warn(
                "Called 'normalize_tensor' with empty 'normalization attribute'. Returning unchanged tensor.", UserWarning)
            return tensor
        mean, std = self.normalization
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(tensor)

    def denormalize_tensor(self, tensor: torch.Tensor, clone: bool = False) -> torch.Tensor:
        if not self.normalization:
            warnings.warn(
                "Called 'denormalize_tensor' with empty 'normalization' attribute. Returning unchanged tensor.", UserWarning)
            return tensor
        mean, std = self.normalization
        tensor = tensor.clone() if clone else tensor
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    def get_features(self, image: torch.Tensor, layers: List[str]) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        model = self.get_model_module()
        x = image
        for name, layer in model._modules.items():
            assert layer
            x = layer(x)
            if name in layers:
                features[name] = x
        return features

    def calc_gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        # default implementation should support both CNNs and Transformers
        if tensor.dim() == 4:
            bs, ch, h, w = tensor.size()
            tensor = tensor.view(bs * ch, h * w)
            gm = torch.mm(tensor, tensor.t())
        elif tensor.dim() == 3:
            bs, seq_len, emb_dim = tensor.size()
            tensor = tensor.view(bs, seq_len, emb_dim)
            gm = torch.bmm(tensor, tensor.transpose(1, 2))
        else:
            raise ValueError(
                "Default calc_gram_matrix implementation only supports either 3 dimensions (CNNs: [batch_size, seq_len, embedding_dim]) or 4 dimensions (Transformers: [batch_size, seq_len, embedding_dim] ).")
        return gm

    def calc_content_loss(self, target: torch.Tensor, content_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        target_features = self.get_features(
            target, layers=[self.content_layer])
        return torch.mean((target_features[self.content_layer] - content_features[self.content_layer]) ** 2)

    def calc_style_loss(
        self,
        target: torch.Tensor,
        style_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=target.device)
        target_features = self.get_features(target, layers=self.style_layers)
        for layer in self.style_layers:
            style_gm = self.calc_gram_matrix(style_features[layer])
            target_gm = self.calc_gram_matrix(target_features[layer])
            loss += self.style_weights[layer] * \
                torch.mean((style_gm - target_gm) ** 2)
        return loss

    def forward(
        self,
        target: torch.Tensor,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        content_features: Optional[Dict[str, torch.Tensor]] = None,
        style_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if content_features is None:
            content_features = self.get_features(
                content_image, layers=[self.content_layer])
        if style_features is None:
            style_features = self.get_features(
                style_image, layers=self.style_layers)
        content_loss = self.calc_content_loss(target, content_features)
        style_loss = self.calc_style_loss(
            target, style_features)
        loss = self.content_weight * content_loss + self.style_weight * style_loss
        return loss

    def visualize(self) -> graphviz.Digraph:
        from stylemod.visualization.module import visualize
        return visualize(self)
