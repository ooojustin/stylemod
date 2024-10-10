from stylos.models.model import Model
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

CONVNEXT_TINY = Model(
    name="CONVNEXT_TINY",
    model_fn=convnext_tiny,
    weights=ConvNeXt_Tiny_Weights.DEFAULT,  # type: ignore[assignment]
    content_layer="4",
    style_layers=[
        "0",
        "1",
        "3",
        "4",
        "5"
    ],
    style_weights={
        "0": 1.0,
        "1": 0.8,
        "3": 0.6,
        "4": 0.4,
        "5": 0.2
    }
)
