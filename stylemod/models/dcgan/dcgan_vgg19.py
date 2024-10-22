import torch
from stylemod.core.gan import GANBaseModel
from stylemod.models.dcgan import DCGANGenerator, DCGANDiscriminator
from stylemod.models.vgg19 import VGG19
from typing import Dict, Optional


class DCGAN_VGG19(GANBaseModel):
    """
    Combines DCGAN's generator and discriminator with VGG19 feature extraction
    for style transfer.
    """

    def __init__(
        self,
        vgg: Optional[VGG19] = None,
        ngpu: int = 1,
        nz: int = 100,
        ngf: int = 64,
        ndf: int = 64,
        nc: int = 3,
        learning_rate_g: float = 0.003,
        learning_rate_d: float = 0.003,
        device: torch.device = torch.device("cpu")
    ):
        if vgg is not None:
            self.vgg = vgg
        else:
            self.vgg = VGG19()

        def generator_fn(): return DCGANGenerator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc)
        def discriminator_fn(): return DCGANDiscriminator(ngpu=ngpu, ndf=ndf, nc=nc)

        super().__init__(
            generator_fn=generator_fn,
            discriminator_fn=discriminator_fn,
            learning_rate_g=learning_rate_g,
            learning_rate_d=learning_rate_d,
            latent_dim=nz,
            device=device
        )

    def calc_style_content_loss(self, generated_image, content_image, style_image):
        content_image = self.vgg.normalize_tensor(content_image)
        style_image = self.vgg.normalize_tensor(style_image)
        content_features = self.vgg.get_features(
            content_image, layers=[self.vgg.content_layer])
        style_features = self.vgg.get_features(
            style_image, layers=self.vgg.style_layers)
        generated_features = self.vgg.get_features(
            generated_image, layers=self.vgg.style_layers + [self.vgg.content_layer])
        generated_content_feature = generated_features[self.vgg.content_layer]
        content_feature = content_features[self.vgg.content_layer]
        if generated_content_feature.size() != content_feature.size():
            generated_content_feature = torch.nn.functional.interpolate(
                generated_content_feature, size=content_feature.shape[2:], mode='bilinear', align_corners=False)
        content_loss = torch.mean(
            (generated_content_feature - content_feature) ** 2)
        style_loss = self.vgg.calc_style_loss(generated_image, style_features)
        return content_loss, style_loss

    def train_step(self, real_images: torch.Tensor, style_image: Optional[torch.Tensor] = None) -> Dict[str, float]:
        real_images = real_images.to(self.device)
        assert style_image is not None, "style_image is required for DCGAN_VGG19."

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

        content_loss, style_loss = self.calc_style_content_loss(
            fake_images, real_images, style_image)

        total_loss = g_loss + self.vgg.content_weight * \
            content_loss + self.vgg.style_weight * style_loss
        total_loss.backward()
        self.optimizer_g.step()

        return {"g_loss": g_loss.item(), "d_loss": d_loss.item(), "content_loss": content_loss.item(), "style_loss": style_loss.item()}

    def initialize_module(self) -> None:
        self.model = self.vgg.get_model_module()

    def set_device(self, device: torch.device) -> torch.nn.Module:
        self.vgg.set_device(device)
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        self.device = device
        return self.generator
