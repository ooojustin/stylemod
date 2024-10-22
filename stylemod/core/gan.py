import torch
from stylemod.core.base import BaseModel, DEFAULTS
from stylemod.core.abstract import NormalizationType
from typing import Callable, Optional, Dict, List
from torch import nn
from torch.optim.adam import Adam


class GANBaseModel(BaseModel):
    """
    A generic GAN model base class. This class can be extended for any GAN implementation
    by passing the generator and discriminator architectures via its constructor.
    """

    def __init__(
        self,
        generator_fn: Callable[..., nn.Module],
        discriminator_fn: Callable[..., nn.Module],
        name: str = "",
        content_layer: str = "",
        style_weights: Dict[str, float] = {},
        content_weight: float = DEFAULTS["content_weight"],
        style_weight: float = DEFAULTS["style_weight"],
        learning_rate_g: float = DEFAULTS["learning_rate"],
        learning_rate_d: float = DEFAULTS["learning_rate"],
        latent_dim: int = 100,  # nz
        normalization: Optional[NormalizationType] = None,
        eval_mode: bool = False,
        retain_graph: bool = False,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__(
            model_fn=generator_fn,
            name=name,
            content_layer=content_layer,
            style_weights=style_weights,
            content_weight=content_weight,
            style_weight=style_weight,
            learning_rate=learning_rate_g,
            normalization=normalization,
            eval_mode=eval_mode,
            retain_graph=retain_graph
        )
        self.generator = generator_fn().to(device)
        self.discriminator = discriminator_fn().to(device)
        self.optimizer_g = Adam(
            self.generator.parameters(), lr=learning_rate_g)
        self.optimizer_d = Adam(
            self.discriminator.parameters(), lr=learning_rate_d)
        self.criterion = nn.BCELoss()
        self.latent_dim = latent_dim
        self.device = device

    def get_features(self, image: torch.Tensor, layers: List[str], use_generator: bool = False) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        if not use_generator:
            model = self.discriminator
            x = image
            for name, layer in model._modules.items():
                assert layer
                x = layer(x)
                if name in layers:
                    features[name] = x
        else:
            noise = torch.randn(image.size(
                0), self.latent_dim, 1, 1, device=image.device)
            x = self.forward_generator(noise)
            features["generated_image"] = x
        return features

    def forward_generator(self, input: torch.Tensor) -> torch.Tensor:
        return self.generator(input)

    def forward_discriminator(self, input: torch.Tensor) -> torch.Tensor:
        return self.discriminator(input)

    def calc_generator_loss(self, fake_output: torch.Tensor) -> torch.Tensor:
        real_labels = torch.ones_like(
            fake_output, device=self.device)
        loss = self.criterion(fake_output, real_labels)
        return loss

    def calc_discriminator_loss(self, real_output: torch.Tensor, fake_output: torch.Tensor) -> torch.Tensor:
        real_labels = torch.ones_like(real_output, device=self.device)
        fake_labels = torch.zeros_like(fake_output, device=self.device)
        real_loss = self.criterion(real_output, real_labels)
        fake_loss = self.criterion(fake_output, fake_labels)
        return real_loss + fake_loss

    def train_step(self, real_images: torch.Tensor, style_image: Optional[torch.Tensor] = None) -> Dict[str, float]:
        real_images = real_images.to(self.device)
        if style_image is not None:
            style_image = style_image.to(self.device)

        # train discriminator
        self.optimizer_d.zero_grad()
        real_output = self.forward_discriminator(real_images)
        noise = torch.randn(real_images.size(
            0), self.latent_dim, 1, 1, device=self.device)
        fake_images = self.forward_generator(noise)
        fake_output = self.forward_discriminator(fake_images.detach())
        d_loss = self.calc_discriminator_loss(real_output, fake_output)
        d_loss.backward()
        self.optimizer_d.step()

        # train generator
        self.optimizer_g.zero_grad()
        fake_output = self.forward_discriminator(fake_images)
        g_loss = self.calc_generator_loss(fake_output)
        g_loss.backward()
        self.optimizer_g.step()

        return {"g_loss": g_loss.item(), "d_loss": d_loss.item()}
