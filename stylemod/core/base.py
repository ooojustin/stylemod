import torch
import warnings
import torchvision.transforms as transforms
from stylemod.core.abstract import AbstractBaseModel
from typing import Callable, Dict, List, Tuple, Optional


NormalizationType = Tuple[Tuple[float, float, float],
                          Tuple[float, float, float]]


class BaseModel(AbstractBaseModel):

    def __init__(
        self,
        model_fn: Callable[..., torch.nn.Module],
        weights=None,
        name: str = "",
        content_layer: str = "",
        style_weights: Dict[str, float] = {},
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
