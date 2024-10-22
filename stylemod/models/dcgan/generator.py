import torch
import torch.nn as nn


class DCGANGenerator(nn.Module):
    def __init__(self, ngpu: int, nz: int, ngf: int, nc: int):
        """
        Args:
            ngpu: Number of GPUs available.
            nz: Size of latent vector.
            ngf: Size of feature maps in the generator.
            nc: Number of channels in the output image.
        """
        super(DCGANGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)
