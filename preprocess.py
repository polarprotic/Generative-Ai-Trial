import os
from PIL import Image
import numpy as np
from rembg import remove


# ── Config ────────────────────────────────────────────────────────────────────

INPUT_DIR  = "Dataset"           # your original Cat/Dog/Horse folders
OUTPUT_DIR = "Dataset_Segmented" # clean images saved here

CLASSES = ["Cat", "Dog", "Horse"]


# ── Process ───────────────────────────────────────────────────────────────────

def remove_background(input_path, output_path):
    """
    Removes background from one image and saves it as a clean RGB image.
    Black pixels where background was, animal pixels untouched.
    """
    with open(input_path, "rb") as f:
        input_data = f.read()

    # rembg returns RGBA — alpha channel = 0 where background is
    output_data = remove(input_data)

    img = Image.open(
        __import__("io").BytesIO(output_data)
    ).convert("RGBA")

    # Create black background and paste animal on top using alpha mask
    background = Image.new("RGB", img.size, (0, 0, 0))
    background.paste(img, mask=img.split()[3])  # alpha as mask

    background.save(output_path)


def preprocess_dataset():

    total = 0
    success = 0

    for class_name in CLASSES:

        input_folder  = os.path.join(INPUT_DIR,  class_name)
        output_folder = os.path.join(OUTPUT_DIR, class_name)

        os.makedirs(output_folder, exist_ok=True)

        files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"\nProcessing {class_name} — {len(files)} images")

        for i, file in enumerate(files):

            input_path  = os.path.join(input_folder,  file)

            # Save all as .png to preserve quality
            output_name = os.path.splitext(file)[0] + ".png"
            output_path = os.path.join(output_folder, output_name)

            # Skip already processed images
            if os.path.exists(output_path):
                print(f"  [{i+1}/{len(files)}] Skipping (exists): {file}")
                success += 1
                total   += 1
                continue

            try:
                remove_background(input_path, output_path)
                success += 1
                print(f"  [{i+1}/{len(files)}] Done: {file}")

            except Exception as e:
                print(f"  [{i+1}/{len(files)}] Failed: {file} — {e}")

            total += 1

    print(f"\nFinished! {success}/{total} images processed.")
    print(f"Segmented dataset saved to → {OUTPUT_DIR}/")


if __name__ == "__main__":
    preprocess_dataset()