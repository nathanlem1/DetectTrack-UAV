import os
import cv2
import numpy as np
import torch
from datetime import datetime
from PIL import Image
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.data.data_augment import preproc

# Load YOLOX experiment config
# exp = get_exp(r"../exps/example/custom/yolox_x_weakaug_2048.py", None)
exp = get_exp(r"../exps/example/custom/yolox_x_weakaug_640.py", None)

model = exp.get_model()
model.eval()
# ckpt = torch.load("../pretrained/yolox_best_ckpt_2048.pth", map_location="cpu")
ckpt = torch.load("../pretrained/yolox_best_ckpt_640.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
model = fuse_model(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def vis(img, outputs, ratio, class_names,
        show_box=True, show_label=True, show_score=True,
        selected_class="All"):
    if outputs is None:
        return img

    outputs = outputs.cpu().detach().tolist()
    boxes = [[c / ratio for c in o[0:4]] for o in outputs]
    scores = [o[4] * o[5] for o in outputs]
    cls_ids = [int(o[6]) for o in outputs]

    for box, score, cls_id in zip(boxes, scores, cls_ids):
        if selected_class != "All" and class_names[cls_id] != selected_class:
            continue

        x0, y0, x1, y1 = map(int, box)
        label = ""
        if show_label:
            label += class_names[cls_id]
        if show_score:
            label += f": {score:.2f}"
        if show_box:
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        if label:
            cv2.putText(img, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img


def run_image_inference(img_pil, conf_thresh=0.3, nms_thresh=0.45, test_size=(896, 896),
                        show_box=True, show_label=True, show_score=True, selected_class="All"):
    img = np.array(img_pil)
    img_info = {"raw_img": img, "height": img.shape[0], "width": img.shape[1]}
    img_pre, ratio = preproc(img, test_size)
    img_tensor = torch.tensor(img_pre.copy().astype(np.float32), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        outputs = postprocess(outputs, exp.num_classes, conf_thresh, nms_thresh)

    os.makedirs("outputs/images", exist_ok=True)
    if outputs[0] is None:
        return None

    result = vis(img_info["raw_img"], outputs[0], ratio, exp.class_names,
                 show_box, show_label, show_score, selected_class)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = f"outputs/images/detect_{timestamp}.jpg"
    cv2.imwrite(out_path, result)
    return out_path


def run_video_inference(video_path, conf_thresh=0.3, nms_thresh=0.45, test_size=(896, 896),
                        show_box=True, show_label=True, show_score=True, selected_class="All",
                        progress=None):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = min(int(fps * 10000), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))  # Max 5 sec

    os.makedirs("outputs/videos", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = f"outputs/videos/detect_{timestamp}.mp4"
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        img_pre, ratio = preproc(frame, test_size)
        img_tensor = torch.tensor(img_pre.copy().astype(np.float32), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            outputs = postprocess(outputs, exp.num_classes, conf_thresh, nms_thresh)

        if outputs[0] is not None:
            result = vis(frame, outputs[0], ratio, exp.class_names,
                         show_box, show_label, show_score, selected_class)
        else:
            result = frame
        writer.write(result)

        if progress:
            progress.progress(min(int(100 * frame_count / max_frames), 100))

        frame_count += 1

    cap.release()
    writer.release()
    return out_path
