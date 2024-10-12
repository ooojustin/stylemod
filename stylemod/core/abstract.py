import torch
from abc import ABC, abstractmethod
from typing import Dict, List


class AbstractBaseModel(ABC):

    @abstractmethod
    def initialize_module(self) -> None:
        raise NotImplementedError(
            "Method not implemented: 'initialize_module'")

    @abstractmethod
    def get_model_module(self) -> torch.nn.Module:
        raise NotImplementedError("Method not implemented: 'get_model_module'")

    @abstractmethod
    def eval(self) -> torch.nn.Module:
        raise NotImplementedError("Method not implemented: 'eval'")

    @abstractmethod
    def set_device(self, device: torch.device) -> torch.nn.Module:
        raise NotImplementedError("Method not implemented: 'set_device'")

    @abstractmethod
    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Method not implemented: 'normalize_tensor'")

    @abstractmethod
    def denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Method not implemented: 'denormalize_tensor'")

    @abstractmethod
    def get_features(self, image: torch.Tensor, layers: List[str]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Method not implemented: 'get_features'")

    @abstractmethod
    def calc_gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Method not implemented: 'calc_gram_matrix'")

    @abstractmethod
    def calc_content_loss(self, target: torch.Tensor, content_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError(
            "Method not implemented: 'calc_content_loss'")

    @abstractmethod
    def calc_style_loss(self, target: torch.Tensor, style_features: Dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Method not implemented: 'calc_style_loss'")

    @abstractmethod
    def forward(self, target: torch.Tensor, content_features: Dict[str, torch.Tensor], style_features: Dict[str, torch.Tensor], content_weight: float, style_weight: float) -> torch.Tensor:
        raise NotImplementedError("Method not implemented: 'forward'")
