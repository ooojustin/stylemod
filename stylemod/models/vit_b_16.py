import torch
from stylemod.core.transformer import TransformerBaseModel
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.nn import MultiheadAttention
from torch.utils.hooks import RemovableHandle
from typing import List


class ViT_B_16(TransformerBaseModel):

    def __init__(
        self,
        model_fn=vit_b_16,
        weights=ViT_B_16_Weights.DEFAULT,
        content_layer="5",
        style_weights={
            "1": 1.0,
            "3": 0.8,
            "5": 0.6,
            "7": 0.4,
            "9": 0.2
        },
        normalization=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        eval_mode=False,
        retain_graph=False,
        use_attention=False
    ):
        super().__init__(
            name="ViT_B_16",
            model_fn=model_fn,
            weights=weights,
            content_layer=content_layer,
            style_weights=style_weights,
            normalization=normalization,
            eval_mode=eval_mode,
            retain_graph=retain_graph,
            use_attention=use_attention
        )

    def get_features(self, image, layers):
        features = {}
        model = self.get_model_module()
        x = model._process_input(image)
        for i, block in enumerate(model.encoder.layers):
            x = block(x)
            if str(i) in layers:
                features[str(i)] = x
        return features

    def get_attention(self, image: torch.Tensor) -> torch.Tensor:
        # it's all you need
        model = self.get_model_module()
        maps: List[torch.Tensor] = []

        def fp_hook(module, input, _):
            q, k, _ = input
            weights = torch.matmul(
                q, k.transpose(-2, -1)) / (module.head_dim ** 0.5)
            weights = torch.nn.functional.softmax(weights, dim=-1)
            maps.append(weights)

        hooks: List[RemovableHandle] = []
        for _, layer in enumerate(model.encoder.layers):
            for submodule in layer.modules():
                if not isinstance(submodule, MultiheadAttention):
                    continue
                handle = submodule.register_forward_hook(fp_hook)
                hooks.append(handle)

        _ = model(image)
        for handle in hooks:
            handle.remove()

        return torch.stack(maps)
