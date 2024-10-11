import torch
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Optional

NormalizationType = Tuple[Tuple[float, float, float],
                          Tuple[float, float, float]]


class AbstractBaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def initialize_module(self) -> None:
        """Initialize the torch.nn.Module ('model' attribute) from the model_fn."""
        pass

    @abstractmethod
    def get_features(self, image: torch.Tensor, layers: List[str]) -> Dict[str, torch.Tensor]:
        """Extract features from the image based on the specified layers."""
        pass

    @abstractmethod
    def set_device(self, device: torch.device) -> torch.nn.Module:
        """Set the device for the model (CPU or GPU)."""
        pass

    @abstractmethod
    def eval(self) -> torch.nn.Module:
        """Set the model to evaluation mode."""
        pass

    @abstractmethod
    def get_model_module(self) -> torch.nn.Module:
        """Return the underlying model module."""
        pass

    @abstractmethod
    def gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        """Calculate the gram matrix for the tensor."""
        pass

    @abstractmethod
    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a given tensor using the model-specific normalization values."""
        pass

    @abstractmethod
    def denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize a given tensor by reversing the model-specific normalization values."""
        pass


class BaseModel(AbstractBaseModel):
    """Base class providing default implementations of the abstract methods."""

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
        """Initialize the model using the model_fn and set to eval mode if required."""
        model = self.model_fn(weights=self.weights)

        # NOTE(justin): not all models have a 'features' attribute (e.g., ResNet, ViT)
        if hasattr(model, 'features'):
            model = model.features

        # disable gradient computation
        for param in model.parameters():
            param.requires_grad_(False)

        self.model = model

    def get_model_module(self) -> torch.nn.Module:
        """Return the model, initializing it first if not already done."""
        if self.model is None:
            self.initialize_module()
        assert self.model is not None, "Model initialization failed."
        return self.model

    def set_device(self, device: torch.device) -> torch.nn.Module:
        """Move the model to the specified device."""
        self.model = self.get_model_module().to(device)
        return self.model

    def eval(self) -> torch.nn.Module:
        """Set the model to evaluation mode."""
        model = self.get_model_module()
        self.model = model.eval()
        return self.model

    def get_features(self, image: torch.Tensor, layers: List[str]) -> Dict[str, torch.Tensor]:
        """Extract features from the image at the specified layers."""
        features = {}
        model = self.get_model_module()

        x = image
        for name, layer in model._modules.items():
            assert layer
            x = layer(x)
            if name in layers:
                features[name] = x

        return features

    def gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        """Calculate the gram matrix for the input tensor."""
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        # gram /= h * w  # TODO(justin): make gram matrix normalization optional
        return gram

    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a given tensor using the model-specific normalization values."""
        if not self.normalization:
            return tensor
        mean, std = self.normalization
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(tensor)

    def denormalize_tensor(self, tensor: torch.Tensor, copy: bool = False) -> torch.Tensor:
        """Denormalize a given tensor by reversing the model-specific normalization values."""
        if not self.normalization:
            return tensor
        mean, std = self.normalization
        if copy:
            new_tensor = tensor.clone()
            for t, m, s in zip(new_tensor, mean, std):
                t.mul_(s).add_(m)
            return new_tensor
        else:
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return tensor
