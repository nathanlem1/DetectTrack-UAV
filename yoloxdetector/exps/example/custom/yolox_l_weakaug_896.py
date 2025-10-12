import os
from yolox.exp import Exp as BaseExp
import torch
from yolox.data.datasets import VISDRONE_CLASSES


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        self.exp_name = os.path.basename(__file__).split(".")[0]
        self.class_names = VISDRONE_CLASSES
        self.num_classes = len(self.class_names)  # 10

        # YOLOX-L Model Configuration
        self.depth = 1.0
        self.width = 1.0
        self.input_size = (896, 896)
        self.test_size = (896, 896)
        self.multiscale_range = 0

        # Dataset paths
        self.data_dir = "detectiondatasets/VisDrone2019"
        self.train_ann = "train.json"
        self.val_ann = "val.json"   # Use this for evaluating on val_images.
        # self.val_ann = "test.json"  # Use this for evaluating on test_images.

        # Training Schedule
        self.max_epoch = 100
        self.warmup_epochs = 5
        self.no_aug_epochs = 15
        self.eval_interval = 5

        # Data Loading
        self.batch_size = 32
        self.data_num_workers = 0 # 20, I changed this (from 20 to 0) since it speeds up on my machine!

        # Light (weak) augmentations. Set all of the probs to 0.0 for no augmentation, to 1.0 for strong (full)
        # augmentation
        self.mosaic_prob = 0.5
        self.mixup_prob = 0
        self.hsv_prob = 0.0
        self.flip_prob = 0.25

        # Optimization
        self.basic_lr_per_img = 0.01 / 64.0
        self.seed = 42

        self.resume = False
        self.ckpt = "YOLOX_weights/yolox_l.pth"  # Set self.ckpt = None for training from scratch!

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train_images",
            img_size=self.input_size,
            preproc=None,
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val_images",  # "val_images" (during training) and "test_images" (for evaluation on test_images)
            img_size=self.test_size,
            preproc=None,
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform
        from torch.utils.data.dataloader import default_collate

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val_images",  # "val_images" (during training) and "test_images" (for evaluation on test_images)
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        return torch.utils.data.DataLoader(
        valdataset,
        batch_size=min(batch_size, 8),   # reduced for evaluation only
        shuffle=False,
        num_workers=6,                  
        pin_memory=False,
        collate_fn=default_collate,
    )
