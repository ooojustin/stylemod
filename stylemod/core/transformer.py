import torch
from stylemod.core.base import BaseModel, NormalizationType
from abc import abstractmethod
from typing import Callable, Dict, Optional


class TransformerBaseModel(BaseModel):
    """Base class for Transformer models."""

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
        super().__init__(
            model_fn=model_fn,
            weights=weights,
            name=name,
            content_layer=content_layer,
            style_weights=style_weights,
            normalization=normalization,
            eval_mode=eval_mode,
            retain_graph=retain_graph
        )
        self.style_attention_maps = None

    @abstractmethod
    def extract_attention(self, image: torch.Tensor) -> torch.Tensor:
        """Extract attention maps from the transformer."""
        pass

    def get_style_loss(self, target: torch.Tensor) -> torch.Tensor:
        """Calculate style loss using attention maps for Transformer models."""
        assert self.style_attention_maps is not None, "Style attention maps must be precomputed. (model.precompute_style_attention)"
        target_attention = self.extract_attention(target)
        style_loss = torch.tensor(0.0, device=target.device)
        for layer in self.style_layers:
            target_gram = self.gram_matrix(target_attention[int(layer)])
            style_gram = self.gram_matrix(
                self.style_attention_maps[int(layer)])
            style_loss += self.style_weights[layer] * \
                torch.mean((target_gram - style_gram) ** 2)
        return style_loss

    def precompute_style_attention(self, style_image: torch.Tensor):
        """Precompute and store attention maps for the style image."""
        self.style_attention_maps = self.extract_attention(style_image)
