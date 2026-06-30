from copy import deepcopy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn.functional as F

from thop import profile

try:
    import torchreid
except ImportError:
    sys.exit("torchreid not found. Install with: pip install torchreid.")


def get_model_info(model, tsize):
    img = torch.zeros((1, 3, tsize[0], tsize[1]), device=next(model.parameters()).device)
    macs, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6  # Number of parameters (in millions)
    macs /= 1e9   # MACs (Multiply-ACcumulate operations)
    flops = macs * 2  # Gflops - Giga FLOPs (Floating Point OPerations). Each MAC counts as two FLOPs.
    # info = "Params: {:f}M, Gflops: {:f}, Gmacs: {:.f}".format(params, flops, macs)
    info = "Params: {:.4f}M, Gflops: {:.4f}, Gmacs: {:.4f}".format(params, flops, macs)
    return info


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


def preprocess(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size) * 114
    img = np.array(image)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r


def load_osnet_ibn(weights_path: str, device: torch.device):
    """
    Build OSNet-IBN from torchreid and load checkpoint weights.

    Handles two checkpoint formats:
      Format 1 — torchreid training checkpoint:
                 {"state_dict": {...}, "epoch": N, ...}
      Format 2 — raw state dict (pretrained weights):
                 {"layer.weight": tensor, ...}

    Read num_classes from classifier.weight shape to avoid size mismatch.
    Strips the 'module.' prefix added by DataParallel if present.
    """
    print(f"[Model] Loading: {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Detect checkpoint (ckpt) format
    if isinstance(ckpt, dict) and "state_dict" in ckpt:  # for .pth.tar ckpt
        state_dict = ckpt["state_dict"]
        epoch = ckpt.get("epoch", "?")
    else:
        # Raw state dict — no wrapper dict
        state_dict = ckpt  # for .pth ckpt
        epoch = "pretrained"

    # Strip 'module.' prefix from DataParallel checkpoints
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        cleaned[new_key] = v

    # Read num_classes from classifier shape
    clf_key = "classifier.weight"
    if clf_key not in cleaned:
        raise KeyError(
            f"Cannot find '{clf_key}' in checkpoint. "
            f"Available keys (first 10): {list(cleaned.keys())[:10]}"
        )
    num_classes = cleaned["classifier.weight"].shape[0]
    print(f"[Model] Building osnet_ibn_x1_0 with num_classes={num_classes}")

    model = torchreid.models.build_model(
        name="osnet_ibn_x1_0", num_classes=num_classes, pretrained=False
    )
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    model.to(device)
    print(f"[Model] Loaded - Epoch: {epoch}")
    return model


class OSNetReIDInterface:
    def __init__(self, weights_path, device, batch_size=8, input_test_size=(256, 128)):
        super(OSNetReIDInterface, self).__init__()
        if device != 'cpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.batch_size = batch_size
        self.input_test_size = list(input_test_size)
        self.pH, self.pW = self.input_test_size

        self.model = load_osnet_ibn(weights_path, torch.device(self.device))
        self.model_info = get_model_info(self.model.to(self.device), tsize=(self.pW, self.pH))

    def inference(self, image, detections):

        if detections is None or np.size(detections) == 0:
            return []

        H, W, _ = np.shape(image)

        batch_patches = []
        patches = []
        for d in range(np.size(detections, 0)):
            tlbr = detections[d, :4].astype(np.int_)
            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])
            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]

            # the model expects RGB inputs
            patch = patch[:, :, ::-1]

            # Apply pre-processing to image.
            patch = cv2.resize(patch, tuple(self.input_test_size[::-1]), interpolation=cv2.INTER_LINEAR)
            # patch, scale = preprocess(patch, self.input_test_size[::-1])

            # plt.figure()
            # plt.imshow(patch)
            # plt.show()

            # Make shape with a new batch dimension which is adapted for network input
            patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
            patch = patch.to(device=self.device)  #.half()

            patches.append(patch)

            if (d + 1) % self.batch_size == 0:
                patches = torch.stack(patches, dim=0)
                batch_patches.append(patches)
                patches = []

        if len(patches):
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)

        features = np.zeros((0, 512))

        for patches in batch_patches:

            # Run model
            patches_ = torch.clone(patches)
            pred = self.model(patches)
            pred[torch.isinf(pred)] = 1.0

            feat = postprocess(pred)

            nans = np.isnan(np.sum(feat, axis=1))
            if np.isnan(feat).any():
                for n in range(np.size(nans)):
                    if nans[n]:
                        # patch_np = patches[n, ...].squeeze().transpose(1, 2, 0).cpu().numpy()
                        patch_np = patches_[n, ...]
                        patch_np_ = torch.unsqueeze(patch_np, 0)
                        pred_ = self.model(patch_np_)

                        patch_np = torch.squeeze(patch_np).cpu()
                        patch_np = torch.permute(patch_np, (1, 2, 0)).int()
                        patch_np = patch_np.numpy()

                        plt.figure()
                        plt.imshow(patch_np)
                        plt.show()

            features = np.vstack((features, feat))

        return features

