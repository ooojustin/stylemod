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
