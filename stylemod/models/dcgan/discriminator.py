import torch
import torch.nn as nn


class DCGANDiscriminator(nn.Module):
    def __init__(self, ngpu: int, ndf: int, nc: int):
        """
        Args:
            ngpu: Number of GPUs available.
            ndf: Size of feature maps in the discriminator.
            nc: Number of channels in the input image.
        """
        super(DCGANDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input).view(-1, 1).squeeze(1)
