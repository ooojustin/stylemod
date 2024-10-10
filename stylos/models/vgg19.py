from stylos.models.model import Model
from torchvision.models import vgg19, VGG19_Weights


VGG19 = Model(
    name="VGG19",
    model_fn=vgg19,
    weights=VGG19_Weights.DEFAULT,  # type: ignore[assignment]
    content_layer="21",
    style_layers=[
        "0",
        "5",
        "10",
        "19",
        "28"
    ],
    style_weights={
        "0": 1.0,
        "5": 0.8,
        "10": 0.5,
        "19": 0.3,
        "28": 0.1
    }
)
