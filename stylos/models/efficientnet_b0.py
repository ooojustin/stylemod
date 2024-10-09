from stylos.models.model import Model
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


EFFICIENTNET_B0 = Model(
    name="EFFICIENTNET_B0",
    model_fn=efficientnet_b0,
    weights=EfficientNet_B0_Weights.DEFAULT,  # type: ignore[assignment]
    content_layer="6",
    style_layers=["0", "2", "4", "6"],
    style_weights={
        # Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        "0": 1.0,
        # MBConv(Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False), Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False))
        "2": 0.8,
        # MBConv(Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False), Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False))
        "4": 0.5,
        # MBConv(Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False), Conv2d(240, 240, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=240, bias=False))
        "6": 0.3,
        # Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        "8": 0.1
    }
)
