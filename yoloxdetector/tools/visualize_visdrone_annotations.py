import os
import random
import json
import cv2
from matplotlib import pyplot as plt

COCO_PATH = "datasets/VisDrone2019/annotations/train.json"
IMG_DIR = "datasets/VisDrone2019/train_images"

with open(COCO_PATH, "r") as f:
    data = json.load(f)

# Build a map: image_id -> image info
id_to_image = {img["id"]: img for img in data["images"]}

# Build a map: image_id -> list of annotations
image_to_anns = {}
for ann in data["annotations"]:
    image_to_anns.setdefault(ann["image_id"], []).append(ann)

# Build a map: category_id -> name
cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

# Pick N random images to visualize
N = 5
selected_images = random.sample(data["images"], N)

for img_info in selected_images:
    img_path = os.path.join(IMG_DIR, img_info["file_name"])
    img = cv2.imread(img_path)
    if img is None:
        print(f" Could not load image {img_path}")
        continue

    anns = image_to_anns.get(img_info["id"], [])
    for ann in anns:
        x, y, w, h = map(int, ann["bbox"])
        class_name = cat_id_to_name[ann["category_id"]]
        color = (0, 255, 0)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title(img_info["file_name"])
    plt.axis("off")
    plt.show()
