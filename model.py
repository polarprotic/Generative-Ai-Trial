import torch
import torch.nn as nn

# ── New Architectural Feature ─────────────────────────────────────────────
class MinibatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Calculate standard deviation across the batch dimension
        std = torch.std(x, dim=0, unbiased=False, keepdim=True) + 1e-8
        
        # Average the standard deviation down to a single scalar number
        mean_std = torch.mean(std)
        
        # Expand this single number to match the spatial dimensions of the image grid
        batch_size, _, height, width = x.shape
        new_channel = mean_std.expand(batch_size, 1, height, width)
        
        # Glue this new "variance" channel onto the original tensor
        return torch.cat([x, new_channel], dim=1)


class Generator(nn.Module):

    def __init__(self, noise_dim=100):

        super().__init__()

        self.noise_dim = noise_dim

        self.embedding = nn.Embedding(
            num_embeddings=4,      
            embedding_dim=noise_dim
        )
        self.net = nn.Sequential(
            nn.Linear(noise_dim * 2, 512 * 4 * 4),
            nn.LeakyReLU(0.2)
        )

        self.conv_net = nn.Sequential(

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, labels, noise):

        emb = self.embedding(labels)
        x   = torch.cat([noise, emb], dim=1)
        x   = self.net(x)
        x   = x.view(-1, 512, 4, 4)
        x   = self.conv_net(x)

        return x


class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(4, 64 * 64)

        self.net = nn.Sequential(

            nn.Conv2d(4, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # ── Architectural Fix Inserted Here ──
            MinibatchStdDev(),

            # ── Note: Input channels changed from 512 to 513 ──
            nn.Conv2d(513, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images, labels):

        emb = self.embedding(labels)
        emb = emb.view(-1, 1, 64, 64)
        x   = torch.cat([images, emb], dim=1)

        return self.net(x).view(-1)





# import torch
# import torch.nn as nn
# from torch.nn.utils import spectral_norm


# NUM_CLASSES = 4


# # ── Minibatch Standard Deviation ──────────────────────────────────────────────
# class MinibatchStdDev(nn.Module):
#     """Appends a channel containing the mean std-dev across the batch.
#     Helps the discriminator detect low-diversity (mode-collapsed) batches."""

#     def forward(self, x):
#         std = torch.std(x, dim=0, unbiased=False, keepdim=True) + 1e-8
#         mean_std = torch.mean(std)
#         b, _, h, w = x.shape
#         new_channel = mean_std.expand(b, 1, h, w)
#         return torch.cat([x, new_channel], dim=1)


# # ── Generator building block: Upsample + Conv (no checkerboard artifacts) ─────
# class UpBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="nearest"),
#             nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#     def forward(self, x):
#         return self.block(x)


# class Generator(nn.Module):

#     def __init__(self, noise_dim=100, num_classes=NUM_CLASSES):
#         super().__init__()
#         self.noise_dim = noise_dim

#         self.embedding = nn.Embedding(num_classes, noise_dim)

#         self.fc = nn.Sequential(
#             nn.Linear(noise_dim * 2, 512 * 4 * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         self.conv_net = nn.Sequential(
#             UpBlock(512, 256),   # 4  → 8
#             UpBlock(256, 128),   # 8  → 16
#             UpBlock(128, 64),    # 16 → 32
#             nn.Upsample(scale_factor=2, mode="nearest"),  # 32 → 64
#             nn.Conv2d(64, 3, 3, 1, 1),
#             nn.Tanh(),
#         )

#     def forward(self, labels, noise):
#         emb = self.embedding(labels)
#         x = torch.cat([noise, emb], dim=1)
#         x = self.fc(x)
#         x = x.view(-1, 512, 4, 4)
#         return self.conv_net(x)


# # ── Projection Discriminator ──────────────────────────────────────────────────
# # - Spectral norm on every conv/linear (stability)
# # - NO BatchNorm in D (it leaks batch statistics)
# # - Label conditioning via projection: out = psi(features) + <embed(y), features>
# # - No Sigmoid: outputs raw logits (used with hinge loss)
# class Discriminator(nn.Module):

#     def __init__(self, num_classes=NUM_CLASSES):
#         super().__init__()

#         self.features = nn.Sequential(
#             spectral_norm(nn.Conv2d(3, 64, 4, 2, 1, bias=False)),   # 64 → 32
#             nn.LeakyReLU(0.2, inplace=True),

#             spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),  # 32 → 16
#             nn.LeakyReLU(0.2, inplace=True),

#             spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)), # 16 → 8
#             nn.LeakyReLU(0.2, inplace=True),

#             spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)), # 8 → 4
#             nn.LeakyReLU(0.2, inplace=True),

#             MinibatchStdDev(),  # 512 → 513 channels

#             spectral_norm(nn.Conv2d(513, 512, 3, 1, 1, bias=False)),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         # Global-sum-pooled feature vector → scalar logit
#         self.linear = spectral_norm(nn.Linear(512, 1))

#         # Projection embedding for class conditioning
#         self.embed = spectral_norm(nn.Embedding(num_classes, 512))

#     def forward(self, images, labels):
#         h = self.features(images)          # (B, 512, 4, 4)
#         h = torch.sum(h, dim=(2, 3))       # global sum pool → (B, 512)

#         out = self.linear(h).squeeze(1)    # unconditional logit
#         emb = self.embed(labels)           # (B, 512)
#         out = out + torch.sum(emb * h, dim=1)  # projection term

#         return out  # raw logits (no sigmoid)