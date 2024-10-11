from stylemod.models.base_model import BaseModel
from torchvision.models import swin_t, Swin_T_Weights


class Swin_T(BaseModel):

    def __init__(self):

        super().__init__(
            model_fn=swin_t,
            weights=Swin_T_Weights.DEFAULT,
            name="Swin_T",
            content_layer="4",
            style_weights={
                "0": 1.0,
                "1": 0.8,
                "2": 0.6,
                "3": 0.4,
                "4": 0.2
            },
            eval_mode=False,
            retain_graph=False
        )
