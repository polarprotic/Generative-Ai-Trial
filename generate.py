import torch
import matplotlib.pyplot as plt

from model import Generator


# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

NOISE_DIM = 100

classes = {
    0: "Cat",
    1: "Dog",
    2: "Horse"
}


# ── Load Model ────────────────────────────────────────────────────────────────

model = Generator(noise_dim=NOISE_DIM).to(device)

model.load_state_dict(
    torch.load(
        "generator.pth",
        map_location=device
    )
)

model.eval()


# ── Input ─────────────────────────────────────────────────────────────────────

label = int(
    input("Enter Class (0=Cat, 1=Dog, 2=Horse): ")
)

num_samples = int(
    input("How many images to generate? (e.g. 4): ")
)

assert label in classes, "Invalid class! Choose 0, 1, or 2."


# ── Generate ──────────────────────────────────────────────────────────────────

with torch.no_grad():

    labels = torch.tensor(
        [label] * num_samples,
        dtype=torch.long
    ).to(device)

    # Different noise → different output each time
    noise = torch.randn(
        num_samples,
        NOISE_DIM,
        device=device
    )

    images = model(labels, noise)   # (N, 3, 64, 64)

    # Denormalize from [-1, 1] back to [0, 1]
    images = (images + 1) / 2
    images = images.clamp(0, 1)


# ── Plot ──────────────────────────────────────────────────────────────────────

cols = min(num_samples, 4)
rows = (num_samples + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

# Flatten axes for easy iteration
if num_samples == 1:
    axes = [axes]
else:
    axes = axes.flatten() if rows > 1 else list(axes)

for i in range(num_samples):

    img = images[i].permute(1, 2, 0).cpu().numpy()

    axes[i].imshow(img)
    axes[i].set_title(classes[label])
    axes[i].axis("off")

# Hide unused subplots
for j in range(num_samples, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(f"generated_{classes[label].lower()}.png", dpi=150)
plt.show()

print(f"Saved → generated_{classes[label].lower()}.png")