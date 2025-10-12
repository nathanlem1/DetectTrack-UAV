import os
import json
import glob
import cv2
from tqdm import tqdm

print(" Starting VisDrone to COCO conversion...")

# 10 valid VisDrone categories 
CATEGORIES = [
    {"id": i, "name": name}
    for i, name in enumerate([
        "pedestrian", "people", "bicycle", "car", "van",
        "truck", "tricycle", "awning-tricycle", "bus", "motor"
    ])
]


def convert_visdrone_to_coco(img_dir, ann_dir, save_path):
    image_id = 0
    annotation_id = 0
    coco = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    for img_path in tqdm(img_files, desc=f"Processing {save_path}"):
        file_name = os.path.basename(img_path)

        # Load image size
        img = cv2.imread(img_path)
        if img is None:
            print(f" Could not read image: {img_path}. Skipping.")
            continue
        height, width = img.shape[:2]

        coco["images"].append({
            "id": image_id,
            "file_name": file_name,  
            "height": height,
            "width": width,
        })

        # Read annotation
        txt_path = os.path.join(ann_dir, file_name.replace(".jpg", ".txt"))
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 6:
                        continue
                    x, y, w, h = map(int, parts[:4])
                    class_id = int(parts[5])

                    if class_id == 0 or class_id > 10:
                        continue  

                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id - 1,  
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    })
                    annotation_id += 1

        image_id += 1

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(coco, f)
    print(f"Saved COCO annotation to: {save_path}")


if __name__ == "__main__":
    base = "./detectiondatasets/VisDrone2019"

    convert_visdrone_to_coco(
        img_dir=f"{base}/train_images",
        ann_dir=f"{base}/annotations/train",
        save_path=f"{base}/annotations/train.json"
    )

    convert_visdrone_to_coco(
        img_dir=f"{base}/val_images",
        ann_dir=f"{base}/annotations/val",
        save_path=f"{base}/annotations/val.json"
    )

    convert_visdrone_to_coco(
        img_dir=f"{base}/test_images",
        ann_dir=f"{base}/annotations/test",
        save_path=f"{base}/annotations/test.json"
    )

    print(" Done! All annotations are converted.")
