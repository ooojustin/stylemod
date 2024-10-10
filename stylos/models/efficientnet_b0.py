from stylos.models.model import Model
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


EFFICIENTNET_B0 = Model(
    name="EFFICIENTNET_B0",
    model_fn=efficientnet_b0,
    weights=EfficientNet_B0_Weights.DEFAULT,  # type: ignore[assignment]
    content_layer="6",
    style_layers=[
        "0",
        "2",
        "4",
        "6"
    ],
    style_weights={
        "0": 1.0,
        "2": 0.8,
        "4": 0.5,
        "6": 0.3,
        "8": 0.1
    }
)
