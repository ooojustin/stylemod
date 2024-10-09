from stylos.models.model import Model
from torchvision.models import vgg19, VGG19_Weights


VGG19 = Model(
    name="VGG19",
    model_fn=vgg19,
    weights=VGG19_Weights.DEFAULT,  # type: ignore[assignment]
    content_layer="21",
    style_layers=["0", "5", "10", "19", "28"],
    style_weights={
        # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        "0": 1.0,
        # Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        "5": 0.8,
        # Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        "10": 0.5,
        # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        "19": 0.3,
        # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        "28": 0.1
    }
)
