import os
import io
from PIL import Image
from rembg import remove


# ── Config ────────────────────────────────────────────────────────────────────

INPUT_DIR  = "Dataset"
OUTPUT_DIR = "Dataset_Segmented"

# Only process Human — Cat/Dog/Horse already done
CLASSES = ["Human"]


# ── Process ───────────────────────────────────────────────────────────────────

def remove_background(input_path, output_path):

    with open(input_path, "rb") as f:
        input_data = f.read()

    output_data = remove(input_data)

    img = Image.open(io.BytesIO(output_data)).convert("RGBA")

    background = Image.new("RGB", img.size, (0, 0, 0))
    background.paste(img, mask=img.split()[3])

    background.save(output_path)


def preprocess_dataset():

    total = 0
    success = 0

    for class_name in CLASSES:

        input_folder  = os.path.join(INPUT_DIR,  class_name)
        output_folder = os.path.join(OUTPUT_DIR, class_name)

        if not os.path.exists(input_folder):
            print(f"❌ Folder not found: {input_folder}")
            continue

        os.makedirs(output_folder, exist_ok=True)

        files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"\nProcessing {class_name} — {len(files)} images")

        for i, file in enumerate(files):

            input_path  = os.path.join(input_folder, file)
            output_name = os.path.splitext(file)[0] + ".png"
            output_path = os.path.join(output_folder, output_name)

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
    print(f"Saved to → {OUTPUT_DIR}/Human/")


if __name__ == "__main__":
    preprocess_dataset()