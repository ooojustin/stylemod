import torch
from abc import ABC, abstractmethod
from typing import Dict, List


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
