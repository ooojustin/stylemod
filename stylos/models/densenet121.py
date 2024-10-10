from stylos.models.model import Model
from torchvision.models import densenet121, DenseNet121_Weights


DENSENET121 = Model(
    name="DENSENET121",
    model_fn=densenet121,
    weights=DenseNet121_Weights.DEFAULT,  # type: ignore[assignment]
    content_layer="denseblock4",
    style_layers=[
        "conv0",
        "denseblock1",
        "denseblock2",
        "denseblock3",
        "denseblock4"
    ],
    style_weights={
        "conv0": 1.0,
        "denseblock1": 0.8,
        "denseblock2": 0.6,
        "denseblock3": 0.4,
        "denseblock4": 0.2
    }
)
