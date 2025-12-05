import os
import cv2
import yaml
import numpy as np

# Load dataset config
with open("config/data_config.yml", "r") as f:
    config = yaml.safe_load(f)

splits = {
    "train": config["train"],
    "test": config["test"],
    "val": config["val"]
}

def generate_segmentation_label(img_path, out_path):
    img = cv2.imread(img_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Range
    lower = np.array([25, 10, 10])
    upper = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Cleanup
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("NO contour found for", img_path)
        return

    cnt = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(cnt)

    epsilon = 0.003 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    h, w = img.shape[:2]

    with open(out_path, "w") as f:
        f.write("0 ")
        for p in approx:
            x = p[0][0] / w
            y = p[0][1] / h
            f.write(f"{x:.6f} {y:.6f} ")
        f.write("\n")


def process_split(base_folder):

    img_folder = os.path.join(base_folder, "images")
    seg_folder = os.path.join(base_folder, "labels_segmentation")

    os.makedirs(seg_folder, exist_ok=True)

    for fname in os.listdir(img_folder):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_folder, fname)
        out_path = os.path.join(seg_folder, fname.replace(".jpg", ".txt").replace(".png", ".txt"))

        generate_segmentation_label(img_path, out_path)


if __name__ == "__main__":
    for split_name, path in splits.items():
        print(f"Processing split: {split_name}")
        process_split(path)
