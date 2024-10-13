import torch
from stylemod.core.transformer import TransformerBaseModel
from stylemod.visualization.gv import noviz
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.hooks import RemovableHandle
from torch.nn import MultiheadAttention
from typing import List, Dict
import ot


@noviz
class ViT_B_16_Sinkhorn(TransformerBaseModel):

    def __init__(self):
        super().__init__(
            model_fn=vit_b_16,
            weights=ViT_B_16_Weights.DEFAULT,
            name="ViT_B_16_Sinkhorn",
            content_layer="5",
            style_weights={
                "1": 1.0,
                "3": 0.8,
                "5": 0.6,
                "7": 0.4,
                "9": 0.2,
                "11": 0.1
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

    def get_attention(self, image: torch.Tensor) -> torch.Tensor:
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

    def calc_transport_map(self, content_features: torch.Tensor, style_features: torch.Tensor, reg: float = 0.05) -> torch.Tensor:
        # optimal transport with entropic regularization
        cf = content_features.view(content_features.size(0), -1)
        sf = style_features.view(style_features.size(0), -1)
        cf_np = cf.detach().cpu().numpy()
        sf_np = sf.detach().cpu().numpy()
        bs = cf_np.shape[0]
        c_unif = ot.unif(bs)
        s_unif = ot.unif(bs)
        cost = ot.dist(cf_np, sf_np, metric="euclidean")
        # T_star = min_T (sum_{i,j} (T_ij * C_ij) + epsilon * sum_{i,j} (T_ij * (log(T_ij) - 1)))
        tmap_np = ot.sinkhorn(c_unif, s_unif, cost, reg)
        tmap = torch.from_numpy(
            tmap_np).to(content_features.device)
        return tmap

    def calc_style_loss(self, target: torch.Tensor, content_features: Dict[str, torch.Tensor], style_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.style_attention is not None, "Style attention maps must be precomputed."
        target_attention = self.get_attention(target)
        loss = torch.tensor(0.0, device=target.device)
        for layer in self.style_layers:
            # target_gm = self.calc_gram_matrix(target_attention[int(layer)])
            target_feature = self.get_features(target, layers=[layer])[layer]
            # style_gm = self.calc_gram_matrix(
            #     self.style_attention[int(layer)])
            transport_map = self.calc_transport_map(
                content_features=target_feature, style_features=style_features[layer])
            transport_loss = torch.mean((target_feature - transport_map) ** 2)
            loss += self.style_weights[layer] * transport_loss
        return loss

    def forward(
        self,
        target: torch.Tensor,
        content_features: Dict[str, torch.Tensor],
        style_features: Dict[str, torch.Tensor],
        content_weight: float,
        style_weight: float
    ) -> torch.Tensor:
        content_loss = self.calc_content_loss(target, content_features)
        style_loss = self.calc_style_loss(
            target, content_features, style_features)
        total_loss = content_weight * content_loss + style_weight * style_loss
        return total_loss
