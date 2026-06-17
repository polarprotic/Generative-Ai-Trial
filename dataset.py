from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms


class AnimalDataset(Dataset):

    def __init__(self, root_dir):

        # Normalize to [-1, 1] to match Tanh output
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),   # data augmentation
            transforms.ColorJitter(              # data augmentation
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(               # [-1, 1] range
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        self.samples = []

        self.class_map = {
            "Cat": 0,
            "Dog": 1,
            "Horse": 2
        }

        for class_name, label in self.class_map.items():

            folder = os.path.join(root_dir, class_name)

            if not os.path.exists(folder):
                print(f"Warning: folder not found: {folder}")
                continue

            count = 0
            for file in os.listdir(folder):

                if file.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    self.samples.append(
                        (
                            os.path.join(folder, file),
                            label
                        )
                    )
                    count += 1

            print(f"Loaded {count} images for class '{class_name}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return torch.tensor(label, dtype=torch.long), image