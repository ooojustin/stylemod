from stylos.models.model import Model
from torchvision.models import swin_t, Swin_T_Weights


SWIN_T = Model(
    name="SWIN_T",
    model_fn=swin_t,
    weights=Swin_T_Weights.DEFAULT,  # type: ignore[assignment]
    content_layer="4",
    style_layers=[
        "0",
        "1",
        "2",
        "3",
        "4"
    ],
    style_weights={
        "0": 1.0,
        "1": 0.8,
        "2": 0.6,
        "3": 0.4,
        "4": 0.2
    }
)
