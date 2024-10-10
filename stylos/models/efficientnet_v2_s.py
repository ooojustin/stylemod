from stylos.models.model import Model
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


EFFICIENTNET_V2 = Model(
    name="EFFICIENTNET_V2_S",
    model_fn=efficientnet_v2_s,
    weights=EfficientNet_V2_S_Weights.DEFAULT,  # type: ignore[assignment]
    content_layer="4",
    style_layers=[
        "0",
        "1",
        "2",
        "4",
        "6"
    ],
    style_weights={
        "0": 1.0,
        "1": 0.8,
        "2": 0.6,
        "4": 0.4,
        "6": 0.2
    }
)
