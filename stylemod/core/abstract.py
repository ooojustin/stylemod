import torch
from abc import ABC, abstractmethod
from typing import Dict, List


class AbstractBaseModel(ABC):

    @abstractmethod
    def initialize_module(self) -> None:
        raise NotImplementedError(
            "Method not implemented: 'initialize_module'")

    @abstractmethod
    def get_features(self, image: torch.Tensor, layers: List[str]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Method not implemented: 'get_features'")

    @abstractmethod
    def set_device(self, device: torch.device) -> torch.nn.Module:
        raise NotImplementedError("Method not implemented: 'set_device'")

    @abstractmethod
    def eval(self) -> torch.nn.Module:
        raise NotImplementedError("Method not implemented: 'eval'")

    @abstractmethod
    def get_model_module(self) -> torch.nn.Module:
        raise NotImplementedError("Method not implemented: 'get_model_module'")

    @abstractmethod
    def calc_gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Method not implemented: 'calc_gram_matrix'")

    @abstractmethod
    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Method not implemented: 'normalize_tensor'")

    @abstractmethod
    def denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Method not implemented: 'denormalize_tensor'")
