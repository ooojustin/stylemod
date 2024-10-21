import torch
import warnings
from stylemod import utils
from stylemod.core.base import BaseModel, NormalizationType, DEFAULTS
from typing import Callable, Dict, Optional


class TransformerBaseModel(BaseModel):
    """
    Eextends BaseModel to implement style transfer using transformers. 
    It introduces attention based style loss calculations and requires computation of attention maps for style transfer tasks.
    Requires an implementation of get_attention() due to the variance in attention mechanisms across transformers.
    """

    # NOTE(justin): Transformers generally perform worse than CNNs on NST tasks.
    # Need to do more research. StyTr2 is an interesting model/paper to refer to: https://arxiv.org/abs/2105.14576
    def __init__(
        self,
        model_fn: Callable[..., torch.nn.Module],
        weights=None,
        name: str = "",
        content_layer: str = "",
        style_weights: Dict[str, float] = {},
        content_weight: float = DEFAULTS["content_weight"],
        style_weight: float = DEFAULTS["style_weight"],
        learning_rate: float = DEFAULTS["learning_rate"],
        normalization: Optional[NormalizationType] = None,
        eval_mode: bool = False,
        retain_graph: bool = False,
        use_attention: bool = False
    ):
        super().__init__(
            model_fn=model_fn,
            weights=weights,
            name=name,
            content_layer=content_layer,
            style_weights=style_weights,
            content_weight=content_weight,
            style_weight=style_weight,
            learning_rate=learning_rate,
            normalization=normalization,
            eval_mode=eval_mode,
            retain_graph=retain_graph
        )
        self.style_attention = None
        self.use_attention = use_attention
        if use_attention and not utils.is_implemented(self, "get_attention"):
            self.use_attention = False
            msg = """
            Initialized transformer based model with 'use_attention = True', but the get_attention() method is not implemented.
            The default loss calculation approach will be used.
            """
            warnings.warn(msg, UserWarning)

    def get_attention(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Method not implemented: 'get_attention'")

    def calc_style_loss(
        self,
        target: torch.Tensor,
        style_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Overrides the base class's style loss calculation to include
        both feature extraction and attention mechanism.
        """
        loss = torch.tensor(0.0, device=target.device)
        target_features = self.get_features(target, layers=self.style_layers)
        for layer in self.style_layers:
            style_gm = self.calc_gram_matrix(style_features[layer])
            target_gm = self.calc_gram_matrix(target_features[layer])
            loss += self.style_weights[layer] * \
                torch.mean((style_gm - target_gm) ** 2)
        if self.use_attention and utils.is_implemented(self, "get_attention"):
            assert self.style_attention is not None, "Style attention not precomputed."
            target_attention = self.get_attention(target)
            for layer in self.style_layers:
                target_att_gm = self.calc_gram_matrix(
                    target_attention[int(layer)])
                style_att_gm = self.calc_gram_matrix(
                    self.style_attention[int(layer)])
                loss += self.style_weights[layer] * \
                    torch.mean((target_att_gm - style_att_gm) ** 2)
        return loss

    def compute_style_attention(self, style_image: torch.Tensor) -> torch.Tensor:
        self.style_attention = self.get_attention(style_image)
        return self.style_attention
