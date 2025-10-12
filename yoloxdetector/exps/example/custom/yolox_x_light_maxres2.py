import os
from yolox.exp import Exp as BaseExp
import torch


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

        self.data_dir = "detectiondatasets/VisDrone2019"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.num_classes = 10

        self.depth = 1.33
        self.width = 1.25

        self.input_size = (2048, 2048)
        self.test_size = (2048, 2048)
        self.multiscale_range = 0

        self.max_epoch = 100
        self.warmup_epochs = 5
        self.no_aug_epochs = 15
        self.eval_interval = 10

        self.batch_size = 32
        self.data_num_workers = 8

        # No augmentations
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.0
        self.enable_mixup = False

        self.basic_lr_per_img = 0.01 / 64.0
        self.seed = 42
        self.resume = False

        # Load pretrained YOLOX-X weights
        self.ckpt = "YOLOX_weights/yolox_x.pth"

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
            cache=False,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val_images",
            img_size=self.test_size,
            preproc=None,
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform
        from torch.utils.data.dataloader import default_collate

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val_images",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        return torch.utils.data.DataLoader(
            valdataset,
            batch_size=min(batch_size, 8),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=default_collate,
        )
