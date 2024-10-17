import torch
from stylemod.core.base import BaseModel, NormalizationType
from typing import Callable, Dict, Optional


class CNNBaseModel(BaseModel):
    """
    Extends BaseModel to implement content and style loss calculations specific to CNN architectures.
    It handles feature extraction and gram matrix computations for style transfer tasks.
    """

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

    def calc_content_loss(self, target: torch.Tensor, content_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        target_features = self.get_features(
            target, layers=[self.content_layer])
        return torch.mean((target_features[self.content_layer] - content_features[self.content_layer]) ** 2)

    def calc_style_loss(
        self,
        target: torch.Tensor,
        style_features: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=device)
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
            target, style_features, target.device)
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        return total_loss
