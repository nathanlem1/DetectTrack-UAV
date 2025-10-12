import os
from yolox.exp import Exp as BaseExp
import torch
from yolox.data.datasets import VISDRONE_CLASSES

torch.cuda.empty_cache()


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # Experiment name based on file name
        self.exp_name = os.path.basename(__file__).split(".")[0]
        self.class_names = VISDRONE_CLASSES

        # Dataset
        self.data_dir = "detectiondatasets/VisDrone2019"
        self.train_ann = "train.json"
        self.val_ann = "val.json"  # Use this for evaluating on val_images.
        # self.val_ann = "test.json"  # Use this for evaluating on test_images.
        self.num_classes = len(self.class_names)  # 10

        # Model: YOLOX-X
        self.depth = 1.33
        self.width = 1.25

        # Default input size
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.multiscale_range = 0

        # Training schedule
        self.max_epoch = 100
        self.warmup_epochs = 5
        self.no_aug_epochs = 15
        self.eval_interval = 10

        # Data loading
        self.batch_size = 2
        self.data_num_workers = 2  

        # Disable all augmentations
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.0
        self.enable_mixup = False

        # Optimization
        self.basic_lr_per_img = 0.01 / 64.0

        # Training from scratch
        self.resume = False
        self.ckpt = None

        # Use mixed precision and no caching to save memory
        self.fp16 = True
        self.cache = False

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
