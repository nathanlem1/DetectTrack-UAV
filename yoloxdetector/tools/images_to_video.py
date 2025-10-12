import cv2
import os
from natsort import natsorted


image_folder = "YOLOX_outputs/yolox_visdrone_s/vis_res/2025_03_29_21_36_46"
output_video = "output_video2.mp4"
fps = 15  

images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png"))]
images = natsorted(images)  # Sort properly (0001, 0002...)

if not images:
    raise ValueError("No images found in folder.")

first_img = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_img.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(img_path)
    out.write(frame)

out.release()
print(f"✅ Video saved as: {output_video}")
