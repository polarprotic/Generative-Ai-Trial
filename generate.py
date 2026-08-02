import torch
import matplotlib.pyplot as plt

from model import Generator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NOISE_DIM = 100


classes = {
    0: "Cat",
    1: "Dog",
    2: "Horse",
    3: "Human"   
}

model = Generator(noise_dim=NOISE_DIM).to(device)
model.load_state_dict(torch.load("generator.pth", map_location=device))
model.eval()

label = int(input("Enter Class (0=Cat, 1=Dog, 2=Horse, 3=Human): "))
num_samples = int(input("How many images to generate? - "))

assert label in classes, "Invalid class! Choose 0, 1, 2, or 3."

with torch.no_grad():

    labels = torch.tensor([label] * num_samples, dtype=torch.long).to(device)
    noise  = torch.randn(num_samples, NOISE_DIM, device=device)
    images = model(labels, noise)

    images = (images + 1) / 2
    images = images.clamp(0, 1)

cols = min(num_samples, 4)
rows = (num_samples + cols - 1) // cols
                               
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

if num_samples == 1:
    axes = [axes]
else:
    axes = axes.flatten() if rows > 1 else list(axes)

for i in range(num_samples):
    img = images[i].permute(1, 2, 0).cpu().numpy()
    axes[i].imshow(img)
    axes[i].set_title(classes[label])
    axes[i].axis("off")

for j in range(num_samples, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(f"generated_{classes[label].lower()}.png", dpi=150)
plt.show()
print(f"Saved → generated_{classes[label].lower()}.png")









# import torch
# import matplotlib.pyplot as plt

# from model import Generator


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NOISE_DIM = 100

# classes = {
#     0: "Cat",
#     1: "Dog",
#     2: "Horse",
#     3: "Human",
# }

# model = Generator(noise_dim=NOISE_DIM).to(device)
# model.load_state_dict(torch.load("generator.pth", map_location=device))
# model.eval()

# label = int(input("Enter Class (0=Cat, 1=Dog, 2=Horse, 3=Human): "))
# num_samples = int(input("How many images to generate? - "))

# assert label in classes, "Invalid class! Choose 0, 1, 2, or 3."

# with torch.no_grad():
#     labels = torch.tensor([label] * num_samples, dtype=torch.long).to(device)
#     noise = torch.randn(num_samples, NOISE_DIM, device=device)

#     # Truncation trick: shrink noise slightly toward 0 for cleaner samples.
#     # 1.0 = off, 0.7 = noticeably cleaner but less diverse.
#     TRUNCATION = 0.8
#     noise = noise * TRUNCATION

#     images = model(labels, noise)
#     images = ((images + 1) / 2).clamp(0, 1)

# cols = min(num_samples, 4)
# rows = (num_samples + cols - 1) // cols

# fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

# if num_samples == 1:
#     axes = [axes]
# else:
#     axes = axes.flatten() if rows > 1 else list(axes)

# for i in range(num_samples):
#     img = images[i].permute(1, 2, 0).cpu().numpy()
#     axes[i].imshow(img)
#     axes[i].set_title(classes[label])
#     axes[i].axis("off")

# for j in range(num_samples, len(axes)):
#     axes[j].set_visible(False)

# plt.tight_layout()
# plt.savefig(f"generated_{classes[label].lower()}.png", dpi=150)
# plt.show()
# print(f"Saved → generated_{classes[label].lower()}.png")