import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import AnimalDataset
from model import Generator, Discriminator


if __name__ == '__main__':

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = AnimalDataset("Dataset_Segmented")
    loader  = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    print(f"Total samples: {len(dataset)}\n")

    # ── Models ────────────────────────────────────────────────────────────────
    NOISE_DIM = 100

    G = Generator(noise_dim=NOISE_DIM).to(device)      # generates fake images
    D = Discriminator().to(device)                      # judges real vs fake

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Binary Cross Entropy — real=1, fake=0
    adversarial_loss = nn.BCELoss()

    # ── Optimizers ────────────────────────────────────────────────────────────
    # Separate optimizers for G and D — they train against each other
    optimizer_G = torch.optim.Adam(
        G.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        D.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999)
    )

    # ── Training ──────────────────────────────────────────────────────────────
    EPOCHS = 100

    print("Starting cGAN training...\n")
    print("What's happening each epoch:")
    print("  D_loss → how well discriminator catches fakes (lower = smarter D)")
    print("  G_loss → how well generator fools discriminator (lower = smarter G)")
    print("  Goal   → both losses should stay balanced around 0.5-0.7\n")

    for epoch in range(EPOCHS):

        running_d_loss = 0
        running_g_loss = 0

        for labels, real_images in loader:

            batch_size  = labels.size(0)
            labels      = labels.to(device)
            real_images = real_images.to(device)

            # Labels for loss — real=1, fake=0
            real_targets = torch.ones(batch_size,  device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            # ── Train Discriminator ───────────────────────────────────────────
            # Goal: correctly identify real images as real, fake images as fake

            optimizer_D.zero_grad()

            # Real images → D should output 1
            d_real_pred = D(real_images, labels)
            d_real_loss = adversarial_loss(d_real_pred, real_targets)

            # Fake images → D should output 0
            noise        = torch.randn(batch_size, NOISE_DIM, device=device)
            fake_images  = G(labels, noise).detach()  # detach so G doesn't update here
            d_fake_pred  = D(fake_images, labels)
            d_fake_loss  = adversarial_loss(d_fake_pred, fake_targets)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # ── Train Generator ───────────────────────────────────────────────
            # Goal: fool the discriminator into thinking fake images are real

            optimizer_G.zero_grad()

            noise       = torch.randn(batch_size, NOISE_DIM, device=device)
            fake_images = G(labels, noise)

            # Generator wants D to output 1 for its fake images
            g_pred = D(fake_images, labels)
            g_loss = adversarial_loss(g_pred, real_targets)

            g_loss.backward()
            optimizer_G.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        n        = len(loader)
        avg_d    = running_d_loss / n
        avg_g    = running_g_loss / n

        # Health check — both losses should stay balanced
        status = "✅ Balanced"
        if avg_d < 0.1:
            status = "⚠️  D too strong — G can't learn"
        elif avg_g < 0.1:
            status = "⚠️  G too strong — D collapsed"

        print(
            f"Epoch [{epoch+1:03d}/{EPOCHS}]  "
            f"D_loss: {avg_d:.4f}  "
            f"G_loss: {avg_g:.4f}  "
            f"{status}"
        )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(G.state_dict(), f"generator_epoch{epoch+1}.pth")
            print(f"  → Saved generator_epoch{epoch+1}.pth\n")

    # ── Save Final ────────────────────────────────────────────────────────────
    torch.save(G.state_dict(), "generator.pth")
    torch.save(D.state_dict(), "discriminator.pth")
    print("\nTraining complete!")
    print("Saved → generator.pth, discriminator.pth")