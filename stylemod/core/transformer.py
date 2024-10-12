import torch
from stylemod.core.base import BaseModel, NormalizationType
from abc import abstractmethod
from typing import Callable, Dict, Optional


class TransformerBaseModel(BaseModel):

    # NOTE(justin): Transformers generally perform worse than CNNs on NST tasks.
    # Need to do more research. StyTr2 is an interesting model/paper to refer to: https://arxiv.org/abs/2105.14576
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
        self.style_attention = None

    @abstractmethod
    def get_attention(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Method not implemented: 'get_attention'")

    def calc_content_loss(self, target: torch.Tensor, content_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        target_features = self.get_features(
            target, layers=[self.content_layer])
        return torch.mean((target_features[self.content_layer] - content_features[self.content_layer]) ** 2)

    def calc_style_loss(self, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert self.style_attention is not None, "Style attention maps must be precomputed. (call model.compute_style_attention())"
        target_attention = self.get_attention(target)
        loss = torch.tensor(0.0, device=target.device)
        for layer in self.style_layers:
            target_gm = self.calc_gram_matrix(target_attention[int(layer)])
            style_gm = self.calc_gram_matrix(self.style_attention[int(layer)])
            loss += self.style_weights[layer] * \
                torch.mean((target_gm - style_gm) ** 2)
        return loss

    def forward(
        self,
        target: torch.Tensor,
        content_features: Dict[str, torch.Tensor],
        style_features: Dict[str, torch.Tensor],
        content_weight: float,
        style_weight: float
    ) -> torch.Tensor:
        content_loss = self.calc_content_loss(target, content_features)
        style_loss = self.calc_style_loss(target)
        total_loss = content_weight * content_loss + style_weight * style_loss
        return total_loss

    def compute_style_attention(self, style_image: torch.Tensor) -> torch.Tensor:
        self.style_attention = self.get_attention(style_image)
        return self.style_attention
