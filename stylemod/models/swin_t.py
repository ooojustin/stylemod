import torch
import torch.nn.functional as F
from stylemod.core.transformer import TransformerBaseModel
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models.swin_transformer import SwinTransformerBlock, ShiftedWindowAttention


class Swin_T(TransformerBaseModel):

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
            normalization=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            eval_mode=False,
            retain_graph=False
        )

    def extract_attention(self, image: torch.Tensor) -> torch.Tensor:
        model = self.get_model_module()
        attention_maps = []

        def hook(module, input, _):
            batch_size, num_windows, num_patches_per_window, channels = input[0].shape

            # get qkv from the input and reshape
            qkv = module.qkv(input[0])
            qkv = qkv.view(batch_size, num_windows * num_patches_per_window,
                           3, module.num_heads, channels // module.num_heads)
            q, k = qkv[:, :, 0], qkv[:, :, 1]

            # compute attention weights
            attn_weights = torch.matmul(
                q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attention_maps.append(attn_weights.detach())

        # register hooks on ShiftedWindowAttention layers
        hooks = []
        for _, stage in model.named_children():
            for block in stage.children():
                if not isinstance(block, SwinTransformerBlock):
                    continue
                if hasattr(block, 'attn') and isinstance(block.attn, ShiftedWindowAttention):
                    hook_handle = block.attn.register_forward_hook(hook)
                    hooks.append(hook_handle)

        # forward pass through the model to trigger hooks, and then remove them
        _ = model(image)
        for hook_handle in hooks:
            hook_handle.remove()

        if len(attention_maps) == 0:
            raise ValueError("No attention maps were extracted.")

        attention_maps_resized = self.resize_attention_maps(attention_maps)
        return torch.stack(attention_maps_resized)

    def resize_attention_maps(self, attention_maps):
        """Resize all attention maps to match the largest spatial and patch size."""
        max_h = max(attn_map.size(-2) for attn_map in attention_maps)
        max_w = max(attn_map.size(-1) for attn_map in attention_maps)
        max_patches = max(attn_map.size(1) for attn_map in attention_maps)

        resized_maps = []
        for attn_map in attention_maps:
            # (batch_size, num_patches, height, width)
            b, n, h, w = attn_map.shape

            # resize the spatial dimensions (height and width)
            # bdd batch dimension for interpolation
            attn_map = attn_map.view(1, n, h, w)
            attn_map_resized = F.interpolate(attn_map, size=(
                max_h, max_w), mode='bilinear', align_corners=False)

            # pad the number of patches (only the patch dimension)
            if n < max_patches:
                padding = (0, 0, 0, 0, 0, max_patches - n)
                attn_map_resized = F.pad(attn_map_resized, padding)

            resized_maps.append(attn_map_resized.view(
                b, max_patches, max_h, max_w))

        return resized_maps
