import torch
from stylemod.core.transformer import TransformerBaseModel
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.nn import MultiheadAttention


class ViT_B_16(TransformerBaseModel):

    def __init__(self):
        super().__init__(
            model_fn=vit_b_16,
            weights=ViT_B_16_Weights.DEFAULT,
            name="ViT_B_16",
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
            retain_graph=False
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

    def extract_attention(self, image: torch.Tensor) -> torch.Tensor:
        model = self.get_model_module()
        attention_maps = []

        def hook(module, input, _):
            # manually compute attention weights
            query, key, _ = input
            attn_weights = torch.matmul(
                query, key.transpose(-2, -1)) / (module.head_dim ** 0.5)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attention_maps.append(attn_weights)

        # register hooks on MultiheadAttention layers
        hooks = []
        for _, layer in enumerate(model.encoder.layers):
            for submodule in layer.modules():
                if not isinstance(submodule, MultiheadAttention):
                    continue
                hook_handle = submodule.register_forward_hook(hook)
                hooks.append(hook_handle)

        # forward pass through the model to trigger hooks, and then remove them
        _ = model(image)
        for hook_handle in hooks:
            hook_handle.remove()

        # stack attention maps into a tensor
        return torch.stack(attention_maps)
