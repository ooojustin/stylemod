import torch
from dataclasses import dataclass
from stylos.models.model import Model
from torchvision.models import vit_b_16, ViT_B_16_Weights


@dataclass
class ViTModel(Model):

    def get_features(self, image, layers):
        """Extract features from transformer blocks in the vision transformer."""
        features = {}
        model = self.get_model_module()
        x = model._process_input(image)

        for i, block in enumerate(model.encoder.layers):
            x = block(x)
            if str(i) in layers:
                features[str(i)] = x

        return features

    def gram_matrix(self, tensor):
        # NOTE(justin): CNNs are 4D tensors, ViTs are 3D tensors
        if tensor.dim() == 4:  # CNN (batch_size, channels, height, width)
            batch_size, d, h, w = tensor.size()
            tensor = tensor.view(batch_size, d, h * w)
        elif tensor.dim() == 3:  # ViT (batch_size, seq_length, hidden_dim)
            batch_size, seq_length, hidden_dim = tensor.size()
            tensor = tensor.view(batch_size, seq_length, hidden_dim)

        gram = torch.bmm(tensor, tensor.transpose(1, 2))
        return gram


VIT_B_16 = ViTModel(
    name="ViT_B_16",
    model_fn=vit_b_16,
    weights=ViT_B_16_Weights.DEFAULT,  # type: ignore[assignment]
    content_layer="5",
    style_layers=[
        "1",
        "3",
        "5",
        "7",
        "9"
    ],
    style_weights={
        "1": 1.0,
        "3": 0.8,
        "5": 0.6,
        "7": 0.4,
        "9": 0.2
    }
)
