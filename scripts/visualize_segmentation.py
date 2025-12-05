import os
import cv2
import random
import numpy as np

def load_segmentation_label(label_file, img_shape):
    """Returns list of segmented polygons."""
    h, w = img_shape[:2]
    polygons = []

    with open(label_file, "r") as f:
        for line in f.readlines():
            values = line.strip().split()
            coords = list(map(float, values[1:]))
            poly_points = []

            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                poly_points.append((x, y))

            polygons.append(poly_points)

    return polygons


def draw_polygons(img, polygons):
    """Draw polygon borders + filled color overlay."""
    overlay = img.copy()

    for poly in polygons:
        pts = np.array(poly, np.int32)

        # Draw outline
        cv2.polylines(img, [pts], True, (0, 255, 0), 3)

        # Fill with transparency
        cv2.fillPoly(overlay, [pts], (0, 255, 0))

    # Blend masks
    img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
    return img


def pick_random_image():
    """Selects a random dataset image from train/test/val."""
    base = "data"

    splits = ["train", "test", "valid"]

    available_paths = []

    for split in splits:
        folder = os.path.join(base, split, "images")
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    available_paths.append((split, f))

    if not available_paths:
        raise Exception("No images found in train/test/valid folders")

    return random.choice(available_paths)


def main():
    split, fname = pick_random_image()

    print(f"Chosen split: {split}")
    print(f"Chosen image: {fname}")

    img_path = os.path.join("data", split, "images", fname)
    seg_label_path = os.path.join(
        "data", split, "labels_segmentation", fname.replace(".jpg", ".txt").replace(".png", ".txt")
    )

    if not os.path.exists(seg_label_path):
        print(f"❌ No segmentation found for {fname}")
        return

    img = cv2.imread(img_path)

    polygons = load_segmentation_label(seg_label_path, img.shape)

    img = draw_polygons(img, polygons)

    cv2.imshow("Random Segmentation", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
