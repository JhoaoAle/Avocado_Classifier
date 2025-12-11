import os
import cv2
from pathlib import Path
import albumentations as A

# ===========================
# PATHS
# ===========================

BASE_PATH = Path(__file__).resolve().parents[1]

ORIG_IMG = BASE_PATH / "data" / "train" / "images"
ORIG_LBL = BASE_PATH / "data" / "train" / "labels"

SYN_IMG = BASE_PATH / "data" / "synthetic" / "images"
SYN_LBL = BASE_PATH / "data" / "synthetic" / "labels"

# ===========================
# CREATE OUTPUT FOLDERS
# ===========================

SYN_IMG.mkdir(parents=True, exist_ok=True)
SYN_LBL.mkdir(parents=True, exist_ok=True)


# ===========================
# AUGMENTATION PIPELINE
# ===========================

augment = A.Compose([

    # Noise
    A.GaussNoise(var_limit=(20.0, 80.0), mean=0, p=0.8),

    # Blur
    A.MotionBlur(blur_limit=15, p=0.6),
    A.GaussianBlur(blur_limit=(3, 7), p=0.7),

    # Light & color variation
    A.RandomBrightnessContrast(p=0.8),
    A.HueSaturationValue(p=0.7),

    # Compression
    A.ImageCompression(quality_range=(10, 40), p=1.0),

    # Downscale
    A.Downscale(scale_range=(0.25, 0.75), p=0.6),

    A.RandomGamma(p=0.6),

    # Random occlusions
    A.CoarseDropout(
        max_holes=8,
        max_height=80,
        max_width=80,
        min_holes=2,
        fill_value=0,
        p=0.5
    )
])

# ===========================
# START GENERATION
# ===========================

image_list = list(ORIG_IMG.glob("*.jpg"))
print(f"Found {len(image_list)} original images.")

counter = 0

for img_path in image_list:

    image = cv2.imread(str(img_path))
    if image is None:
        continue

    # apply noise
    noisy = augment(image=image)['image']

    # synthetic image filename
    new_name = f"synt_{counter}.jpg"

    cv2.imwrite(str(SYN_IMG / new_name), noisy)

    # COPY LABEL WITHOUT MODIFICATION
    original_label_name = img_path.stem + ".txt"
    new_label_name = f"synt_{counter}.txt"

    original_label_path = ORIG_LBL / original_label_name
    new_label_path = SYN_LBL / new_label_name

    if original_label_path.exists():
        with open(original_label_path, "r") as src:
            with open(new_label_path, "w") as dst:
                dst.write(src.read())

    counter += 1

print(f"\nSuccessfully created {counter} noisy synthetic samples")
print("Stored under data/synthetic/")
