import torch
from stylemod.core.base import BaseModel, NormalizationType
from typing import Callable, Dict, Optional


class CNNBaseModel(BaseModel):

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

    def calc_style_loss(
        self,
        target_features: Dict[str, torch.Tensor],
        style_features: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=device)
        for layer in self.style_layers:
            style_gm = self.calc_gram_matrix(style_features[layer])
            target_gm = self.calc_gram_matrix(target_features[layer])
            loss += self.style_weights[layer] * \
                torch.mean((style_gm - target_gm) ** 2)
        return loss
