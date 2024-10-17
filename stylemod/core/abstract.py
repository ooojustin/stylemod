import torch
import graphviz
from abc import ABC, abstractmethod
from typing import Dict, List
from typing import Callable, Tuple, Optional


NormalizationType = Tuple[Tuple[float, float, float],
                          Tuple[float, float, float]]


class AbstractBaseModel(ABC):
    """
    Defines the required interface for all models, ensuring consistency in methods like initialization, feature extraction, and loss calculations. 
    It does not provide any concrete implementations.
    BaseModel can be extended to reuse default implementations for common functionality.
    """

    model_fn: Callable[..., torch.nn.Module]
    weights = None,
    name: str
    content_layer: str
    style_layers: List[str]  # list(style_weights.keys())
    style_weights: Dict[str, float]
    content_weight: float
    style_weight: float
    learning_rate: float
    normalization: Optional[NormalizationType]
    eval_mode: bool = False
    retain_graph: bool = False

    @abstractmethod
    def initialize_module(self) -> None:
        pass

    @abstractmethod
    def get_model_module(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def eval(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def set_device(self, device: torch.device) -> torch.nn.Module:
        pass

    @abstractmethod
    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_features(self, image: torch.Tensor, layers: List[str]) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def calc_gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def calc_content_loss(self, target: torch.Tensor, content_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def calc_style_loss(self, target: torch.Tensor, style_features: Dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, target: torch.Tensor, content_features: Dict[str, torch.Tensor], style_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def visualize(self) -> graphviz.Digraph:
        pass
