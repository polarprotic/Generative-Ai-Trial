import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, noise_dim=100):

        super().__init__()

        self.noise_dim = noise_dim

        # Class embedding: label → 100-dim vector
        self.embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=noise_dim
        )

        # noise_dim + embedding_dim = 200 input features
        self.net = nn.Sequential(

            # 200 → reshape to (512, 4, 4)
            nn.Linear(noise_dim * 2, 512 * 4 * 4),
            nn.LeakyReLU(0.2)
        )

        self.conv_net = nn.Sequential(

            # (512, 4, 4) → (256, 8, 8)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # (256, 8, 8) → (128, 16, 16)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # (128, 16, 16) → (64, 32, 32)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # (64, 32, 32) → (3, 64, 64)
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # outputs in [-1, 1]
        )

    def forward(self, labels, noise):

        # Embed the class label
        emb = self.embedding(labels)  # (B, 100)

        # Concatenate noise + embedding
        x = torch.cat([noise, emb], dim=1)  # (B, 200)

        # MLP head
        x = self.net(x)                     # (B, 512*4*4)
        x = x.view(-1, 512, 4, 4)          # (B, 512, 4, 4)

        # Convolutional upsampling
        x = self.conv_net(x)               # (B, 3, 64, 64)

        return x


class Discriminator(nn.Module):
    """
    Optional discriminator for GAN training.
    Currently unused but included for future upgrade.
    """

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(3, 64 * 64)

        self.net = nn.Sequential(

            # (4, 64, 64) → (64, 32, 32)   (3 img + 1 label channel)
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),

            # (64, 32, 32) → (128, 16, 16)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # (128, 16, 16) → (256, 8, 8)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # (256, 8, 8) → (512, 4, 4)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # (512, 4, 4) → scalar
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images, labels):

        # Embed label as an extra channel
        emb = self.embedding(labels)
        emb = emb.view(-1, 1, 64, 64)

        x = torch.cat([images, emb], dim=1)  # (B, 4, 64, 64)

        return self.net(x).view(-1)