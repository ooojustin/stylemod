import torch
from stylemod.core.cnn import CNNBaseModel
from stylemod.visualization.gv import noviz
from torchvision.models import vgg19, VGG19_Weights
from typing import Dict, Optional


@noviz
class VGG19_EOT(CNNBaseModel):
    # VGG19 using both gram matrix and entropic optimal transport
    # https://www.math.columbia.edu/~mnutz/docs/EOT_lecture_notes.pdf

    def __init__(self, reg: float = 0.003, use_eot_layers: Optional[Dict[str, bool]] = None):
        super().__init__(
            model_fn=vgg19,
            weights=VGG19_Weights.DEFAULT,
            name="VGG19_EOT",
            content_layer="21",
            style_weights={
                "0": 1.0,   # conv1_1
                "5": 0.8,   # conv2_1
                "10": 0.5,  # conv3_1
                "19": 0.3,  # conv4_1
                "28": 0.1   # conv5_1
            },
            normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            eval_mode=False,
            retain_graph=False
        )

        self.reg = reg  # entropic regularization (sinkhorn epsilon)
        self.use_eot_layers = use_eot_layers if use_eot_layers else {
            layer: False for layer in self.style_layers}

        # default eot layers (conv2_1, conv4_1, conv5_1)
        if not use_eot_layers:
            self.use_eot_layers["5"] = True
            self.use_eot_layers["19"] = True
            self.use_eot_layers["28"] = True

    def calc_style_loss(self, target: torch.Tensor, style_features: Dict[str, torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
        loss = torch.tensor(0.0, device=device)
        target_features = self.get_features(target, layers=self.style_layers)

        for layer in self.style_layers:
            # gram matrix loss
            style_gm = self.calc_gram_matrix(style_features[layer])
            target_gm = self.calc_gram_matrix(target_features[layer])
            gram_loss = self.style_weights[layer] * \
                torch.mean((style_gm - target_gm) ** 2)
            loss += gram_loss

            # apply EOT loss if the layer is flagged
            if self.use_eot_layers.get(layer, False):
                eot_loss = self.style_weights[layer] * self.calc_eot_loss(
                    target_features[layer], style_features[layer], device)
                loss += eot_loss

        return loss

    def calc_eot_loss(self, target_features: torch.Tensor, style_features: torch.Tensor, device: Optional[torch.device]) -> torch.Tensor:
        # downsample tensors for testing to avoid memory issues
        # a 400x400 image would result in 160,000 patches
        # (160000 * 160000) * 4 bytes per f32 = ~102 GB of memory to compute?
        sz = (32, 32)
        target_features = torch.nn.functional.interpolate(target_features, sz)
        style_features = torch.nn.functional.interpolate(style_features, sz)
        bs, channels, _, _ = target_features.size()
        patches = sz[0] * sz[1]
        target_features = target_features.view(bs, channels, patches)
        style_features = style_features.view(bs, channels, patches)
        transport_map = self.calc_transport_map(
            target_features, style_features)
        transported_style_features = torch.bmm(
            transport_map, style_features.permute(0, 2, 1)).permute(0, 2, 1)
        transport_loss = torch.mean(
            (target_features - transported_style_features) ** 2)
        return transport_loss

    def calc_transport_map(self, content_features: torch.Tensor, style_features: torch.Tensor) -> torch.Tensor:
        bs, _, patches = content_features.size()
        cost = torch.cdist(content_features.permute(
            0, 2, 1), style_features.permute(0, 2, 1), p=2)
        c_unif = torch.ones(
            (bs, patches), device=content_features.device) / patches
        s_unif = torch.ones(
            (bs, patches), device=style_features.device) / patches
        transport_map = self.sinkhorn(c_unif, s_unif, cost, self.reg)
        return transport_map

    def sinkhorn(self, source: torch.Tensor, target: torch.Tensor, cost: torch.Tensor, reg: float, num_iters: int = 100) -> torch.Tensor:
        # T_star = argmin_T (sum_{i,j} (T_ij * C_ij) + epsilon * sum_{i,j} (T_ij * (log(T_ij) - 1)))
        # where:
        # - T_star is the output transport plan matrix (T) that minimizes the objective func
        # - T_ij is the element at position (i, j) in the transport plan matrix (T), representing the amount of data transported from source i to target j
        # - C_ij is the cost associated with transporting feature data from location i (in the content image) to location j (in the style image)
        # - epsilon is the entropy regularization parameter which will impact the smoothness of the transport plan
        kernel = torch.exp(-cost / reg)
        src = torch.ones_like(source)
        tgt = torch.ones_like(target)
        eps = 1e-6  # tiny minimum epsilon
        for _ in range(num_iters):
            src = source / \
                torch.clamp(kernel @ tgt.unsqueeze(-1), min=eps).squeeze(-1)
            tgt = target / torch.clamp(kernel.transpose(1, 2) @
                                       src.unsqueeze(-1), min=eps).squeeze(-1)
        # diag(src) * kernel * diag(tgt)
        transport_plan = src.unsqueeze(-1) * kernel * tgt.unsqueeze(-2)
        return transport_plan
