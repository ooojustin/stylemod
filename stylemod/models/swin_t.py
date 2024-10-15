import torch
from stylemod.core.transformer import TransformerBaseModel
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models.swin_transformer import SwinTransformerBlock, ShiftedWindowAttention
from torch.utils.hooks import RemovableHandle
from typing import List


class Swin_T(TransformerBaseModel):

    def __init__(
        self,
        model_fn=swin_t,
        weights=Swin_T_Weights.DEFAULT,
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
    ):
        super().__init__(
            name="Swin_T",
            model_fn=model_fn,
            weights=weights,
            content_layer=content_layer,
            style_weights=style_weights,
            normalization=normalization,
            eval_mode=eval_mode,
            retain_graph=retain_graph
        )

    def get_attention(self, image: torch.Tensor) -> torch.Tensor:
        # it's all you need
        model = self.get_model_module()
        maps: List[torch.Tensor] = []

        def fp_hook(module, input, _):
            # batch size, windows, patches per window, channels
            bs, win, ppwin, ch = input[0].shape
            qkv = module.qkv(input[0])
            qkv = qkv.view(bs, win * ppwin,
                           3, module.num_heads, ch // module.num_heads)
            q, k = qkv[:, :, 0], qkv[:, :, 1]
            weights = torch.matmul(
                q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
            weights = torch.softmax(weights, dim=-1)
            maps.append(weights.detach())

        hooks: List[RemovableHandle] = []
        for _, stage in model.named_children():
            for block in stage.children():
                if not isinstance(block, SwinTransformerBlock):
                    continue
                if hasattr(block, "attn") and isinstance(block.attn, ShiftedWindowAttention):
                    handle = block.attn.register_forward_hook(fp_hook)
                    hooks.append(handle)

        _ = model(image)
        for handle in hooks:
            handle.remove()

        maps = self.resize_attention_maps(maps)
        return torch.stack(maps)

    def resize_attention_maps(self, attention_maps):
        max_h = max(attn_map.size(-2) for attn_map in attention_maps)
        max_w = max(attn_map.size(-1) for attn_map in attention_maps)
        max_p = max(attn_map.size(1) for attn_map in attention_maps)
        maps = []
        for attn_map in attention_maps:
            # batch_size, num_patches, height, width
            bs, p, h, w = attn_map.shape
            map = attn_map.view(1, p, h, w)
            map_resized = torch.nn.functional.interpolate(map, size=(
                max_h, max_w), mode="bilinear", align_corners=False)
            if p < max_p:
                padding = (0, 0, 0, 0, 0, max_p - p)
                map_resized = torch.nn.functional.pad(map_resized, padding)
            maps.append(map_resized.view(
                bs, max_p, max_h, max_w))
        return maps
