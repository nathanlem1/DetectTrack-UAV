import os
from yolox.exp import Exp as BaseExp
import torch
from yolox.data.datasets import VISDRONE_CLASSES


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # Set experiment name automatically based on file name
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.class_names = VISDRONE_CLASSES

        # Dataset configuration
        self.data_dir = "detectiondatasets/VisDrone2019"
        self.train_ann = "train.json"
        self.val_ann = "val.json"   # Use this for evaluating on val_images.
        # self.val_ann = "test.json"  # Use this for evaluating on test_images.
        # self.test_ann = "test.json"
        self.num_classes = len(VISDRONE_CLASSES)  # 10

        # Model configuration
        self.depth = 1.33
        self.width = 1.25

        # Input/output resolution
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.multiscale_range = 0  # Disable multiscale augmentation

        # Training schedule
        self.max_epoch = 100
        self.warmup_epochs = 5
        self.no_aug_epochs = 15
        self.eval_interval = 10

        # Data Loading
        self.batch_size = 32
        self.data_num_workers = 0  # 12, I changed this (from 12 to 0) since it speeds up on my machine!

        # Light (weak) augmentations. Set all of the probs to 0.0 for no augmentation, to 1.0 for strong (full)
        # augmentation
        self.mosaic_prob = 0.5
        self.mixup_prob = 0.0
        self.hsv_prob = 0.5
        self.flip_prob = 0.25
        self.enable_mixup = False

        # Optimization
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0  # Todo: Is it possible to make batch_size 64.0 dynamic?
        self.seed = 42
        self.resume = False

        # Load pretrained YOLOX-X weights
        self.ckpt = "YOLOX_weights/yolox_x.pth"  # Set self.ckpt = None for training from scratch!

        # Use mixed precision (fp16)
        self.fp16 = True

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
            name="val_images",   # "val_images" (during training) and "test_images" (for evaluation on test_images)
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        return torch.utils.data.DataLoader(
            valdataset,
            batch_size=min(batch_size, 8),
            shuffle=False,
            num_workers=self.data_num_workers,
            pin_memory=True,
            collate_fn=default_collate,
        )
