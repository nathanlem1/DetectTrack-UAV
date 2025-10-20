# DetectTrack-UAV
This code implements [DetectTrack-UAV: Multi-object Detection and Tracking in Moving UAV Videos]().

## Abstract
In this work, we investigate detection and tracking of multiple objects in moving unmanned aerial vehicle (UAV) videos. 
First, we fine-tune [YOLOX-X](https://github.com/Megvii-BaseDetection/YOLOX) object detector to 
[VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) detection images dataset. After investigating object detector 
performances on the VisDrone2019 images dataset, we integrate it into a Kalman filter (KF) for tracking multiple objects. 
In addition to considering motion as strong cue, we also consider weak cues such as height intersection-over-union 
(height-IoU) and tracklet confidence in the data association using a weighted sum fusion method. We conduct 
extensive evaluations on [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) 
and [UAVDT](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5) Multi-Object Tracking (MOT) datasets as a zero-shot 
solution, and find out that our proposed tracker, DetectTrack-UAV, performs competitively against the existing 
state-of-the-art methods.

The qualitative result (demo) of the DetectTrack-UAV on VisDrone2019-MOT-test-dev data, particularly on a 
uav0000120_04775_v sequence, is shown below. 

![](./assets/uav0000120_04775_v_f189_to_f271.gif)


We show the important steps of this work as follows: 1) Detection and 2) Tracking.

## 1. Detection

### Installation

**Step 1.** Git clone this repo and install it. Check if yours is pip or pip3 and python or python3.
```shell
git clone https://github.com/nathanlem1/DetectTrack-UAV.git
cd DetectTrack-UAV/yoloxdetector/
pip install -r requirements.txt
python setup.py develop  
```
The code was tested using torch 2.2.2+cu118 and torchvision 0.17.2+cu118. You can install torch and matched torchvision 
from [pytorch.org](https://pytorch.org/get-started/locally/).

Please check the [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) README for any installation issues.

**Step 2.** Install [pycocotools](https://github.com/cocodataset/cocoapi).
```shell
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

**Step 3.** Others
```shell
# Cython-bbox
pip install cython_bbox

# faiss cpu / gpu
pip install faiss-cpu
pip install faiss-gpu
```

### Detection Data Preparation

First, we need to convert [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) `Object Detection in Images` 
dataset to COCO format using a script given in `yoloxdetector/tools/visdrone_to_coco.py`. The detection dataset is 
expected to be put in the `DetectTrack-UAV/yoloxdetector/detectiondatasets/VisDrone2019` folder according to the 
following structure:


```
DetectTrack-UAV/yoloxdetector/detectiondatasets/VisDrone2019
                                            |——————Annotations
                                            |        └——————train.json
                                            |        └——————val.json
                                            |        └——————test.json
                                            |——————train_images
                                            |        └—————— 0000002_00005_d_0000014.jpg
                                            |        └——————0000002_00448_d_0000015.jpg
                                            |        └—————— ...
                                            |——————val_images
                                            |        └——————0000001_02999_d_0000005.jpg
                                            |        └——————0000001_03499_d_0000006.jpg
                                            |        └——————...
                                            |——————test_images
                                            |        └——————0000006_00159_d_0000001.jpg
                                            |        └——————0000006_00611_d_0000002.jpg
                                            |        └——————...
```

### Training
Download the weights of the pretrained [YOLOX-X](https://github.com/Megvii-BaseDetection/YOLOX) object detector model to 
the `DetectTrack-UAV/yoloxdetector/YOLOX_weights` directory, which is used for fine-tuning instead of training from scratch.
Then, you can start detection training by running the following command:

```shell
# Being in DetectTrack-UAV/yoloxdetector folder
python tools/train.py -f exps/example/custom/yolox_x_weakaug_640.py -d 1 -b 8 --fp16 -o -c ./YOLOX_weights/yolox_x.pth
``` 
This command uses `640 x 640` input image size.

As an another alternative, we choose the input size by conducting the average of the VisDrone2019 training image sizes 
(1002x1520) and then taking the next 32-divisible number (1024x1536) to align with the 32-divisiblility requirement of 
YOLOX model. You can use this input image size by using `yolox_x_weakaug_1024_1536.py` as follows:

```shell
# Being in DetectTrack-UAV/yoloxdetector folder
python tools/train.py -f exps/example/custom/yolox_x_weakaug_1024_1536.py -d 1 -b 8 --fp16 -o -c ./YOLOX_weights/yolox_x.pth
``` 

Note that increasing the input image size increases the detection performance at the expense of more computation time, 
both training and inference time.

### Detection Demo
First, you need to download a pretrained model from [here](https://drive.google.com/file/d/12BoRMRhfbBHnoN45lyVLQVxIWGLTjFI1/view?usp=drive_link) 
and then put in `DetectTrack-UAV/yoloxdetector/pretrained` folder.

* **Method 1:** Using traditional approach

Demo for image:
```shell
# Being in DetectTrack-UAV/yoloxdetector folder
python tools/demo.py image -f exps/example/custom/yolox_x_weakaug_640.py -c ./pretrained/yolox_best_ckpt_640.pth --path assets/sample_image.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```

Demo for video:
```shell
# Being in DetectTrack-UAV/yoloxdetector folder
python tools/demo.py video -f exps/example/custom/yolox_x_weakaug_640.py -c ./pretrained/yolox_best_ckpt_640.pth --path assets/sample_video.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```


* **Method 2:** Using Streamlit


First you need to install Streamlit app dependencies by changing path:

`cd yolox_streamlit_app` 

and then run the following to install the dependencies:

```shell
# Being in DetectTrack-UAV/yoloxdetector/yolox_streamlit_app folder
pip install -r requirements.txt
```

and then run the following to run demo based on image or video input:

```shell
# Being in DetectTrack-UAV/yoloxdetector/yolox_streamlit_app folder
streamlit run app.py
```
Note that you may need to change the path to the custom exp path in the `yolox_inference.py`, placed 
here: `exp = get_exp(r"../exps/example/custom/yolox_x_weakaug_640.py", None)` and its corresponding ckpt 
`ckpt = torch.load("../pretrained/yolox_best_ckpt_640.pth", map_location="cpu")`.




## 2. Tracking 

### Tracking Data Preparation

Download [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) and [UAVDT](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5) 
Multi-Object Tracking (MOT) datasets. And put them in the `DetectTrack-UAV/trackingdatasets` folder according to the 
following structure:

```
DetectTrack-UAV/trackingdatasets
                    |——————VisDrone2019/MOT
                    |        └——————VisDrone2019-MOT-test-dev
                    |               └——————annotations 
                    |                       └——————uav0000009_03358_v.txt
                    |                       └——————uav0000073_00600_v.txt
                    |                       └——————...
                    |               └——————sequences
                    |                       └——————uav0000009_03358_v 
                    |                               └——————0000001.jpg
                    |                               └——————0000002.jpg
                    |                               └——————...
                    |                       └——————uav0000073_00600_v
                    |                               └——————0000001.jpg
                    |                               └——————0000002.jpg
                    |                               └——————...
                    |                       └——————...
                    |        └——————VisDrone2019-MOT-train
                    |        └——————VisDrone2019-MOT-val
                    |——————UAVDT
                    |        └——————UAV-benchmark-M  
                    |               └——————M0203
                    |                       └——————img000001.jpg
                    |                       └——————img000002.jpg
                    |                       └——————...
                    |               └——————M0205
                    |                       └——————img000001.jpg
                    |                       └——————img000002.jpg
                    |                       └——————...
                    |               └——————...
                    |        └——————UAV-benchmark-MOTD_v1.0
                    |               └——————GT  
                    |                     └——————M0203_gt.txt
                    |                     └——————M0203_ignore.txt
                    |                     └——————M0203_whole.txt        
                    |                     └——————M0203_gt_merge.txt  (after merging M0203_gt.txt and M0203_ignore.txt)
                    |                     └——————...
```

### Pretrained Model 

First, you need to download a pretrained detection model from [here](https://drive.google.com/file/d/12BoRMRhfbBHnoN45lyVLQVxIWGLTjFI1/view?usp=drive_link) 
and then put in `DetectTrack-UAV/yoloxdetector/pretrained` 
folder, and then follow the following instructions.


* **Running on VisDrone2019**

To run the tracker on the VisDrone2019 test dataset for single-class evaluation, `VisDrone2019-MOT-test-dev`, you need to run:

```shell
# Using motion only, Being in 'DetectTrack-UAV/' folder
python run_tracker_uav.py --path ./trackingdatasets/VisDrone2019/MOT/VisDrone2019-MOT-test-dev/sequences --default-parameters --benchmark VisDrone --eval test --experiment-name FSORTuav1 --fp16 --fuse

# Using motion, hiou and confidence distances
python run_tracker_uav.py --path ./trackingdatasets/VisDrone2019/MOT/VisDrone2019-MOT-test-dev/sequences --default-parameters --with-hiou --with-confidence --benchmark VisDrone --eval test --experiment-name FSORTuav1 --fp16 --fuse
```

To run the tracker on the VisDrone2019 test dataset for multi-class evaluation, `VisDrone2019-MOT-test-dev`, you need to run:

```shell
# Using motion only, Being in 'DetectTrack-UAV/' folder
python run_tracker_uav.py --path ./trackingdatasets/VisDrone2019/MOT/VisDrone2019-MOT-test-dev/sequences --default-parameters --benchmark VisDrone --eval test --multi_class_eval --experiment-name FSORTuav2 --fp16 --fuse

# Using motion, hiou and confidence distances
python run_tracker_uav.py --path ./trackingdatasets/VisDrone2019/MOT/VisDrone2019-MOT-test-dev/sequences --default-parameters --with-hiou --with-confidence --benchmark VisDrone --eval test --multi_class_eval --experiment-name FSORTuav2 --fp16 --fuse
```

* **Running on UAVDT**

To run the tracker on the UAVDT test dataset, you need to run:

```shell
# Using motion only, Being in 'DetectTrack-UAV/' folder
python run_tracker_uav.py  --path ./trackingdatasets/UAVDT/UAV-benchmark-M --default-parameters --benchmark UAVDT --eval test --experiment-name FSORTuav1 --fp16 --fuse

# Using motion, hiou and confidence distances
python run_tracker_uav.py --path ./trackingdatasets/UAVDT/UAV-benchmark-M --default-parameters --with-hiou --with-confidence --benchmark UAVDT --eval test --experiment-name FSORTuav1 --fp16 --fuse
```

* **Interpolation**

This is optional. 

For using linear interpolation (LI), run the following code:
```shell
# Being in 'DetectTrack-UAV/' folder
python tools/linear_interpolation.py --txt_path <path_to_track_result>
```

For using Gaussian-smoothed interpolation (GSI), run the following code:
```shell
# Being in 'DetectTrack-UAV/' folder
python tools/gaussian_smoothed_interpolation.py --txt_path <path_to_track_result>
```


### Tracking Evaluation

You can use the MOTChallenge evaluation code from [Easier_To_Use_TrackEval](https://github.com/JackWoo0831/Easier_To_Use_TrackEval) 
to evaluate the tracker's performance on the VisDrone2019 and UAVDT `test` datasets. Hence, to evaluate on VisDrone2019 
and UAVDT datasets, you need to follow the following instructions.

* **Evaluating on VisDrone2019**

Note that VisDrone can be evaluated as a single category or multiple categories.

If it is a single category review, please run the following first to combine the five valid categories specified by 
VisDrone into one valid category for evaluating together. You run this only once.

```bash
# Being in 'DetectTrack-UAV/' folder
python Easier_To_Use_TrackEval/dataset_tools/merge_visdrone_categories.py --data_root ./trackingdatasets/VisDrone2019/MOT/
```

Then, run the following for single class evaluation:

```bash
# Being in 'DetectTrack-UAV/' folder
python Easier_To_Use_TrackEval/scripts/run_custom_dataset.py --config_path Easier_To_Use_TrackEval/configs/VisDrone_test_dev_merge_class.yaml
```

For multi-class evaluation, run the following:

```bash
# Being in 'DetectTrack-UAV/' folder
python Easier_To_Use_TrackEval/scripts/run_custom_dataset.py --config_path Easier_To_Use_TrackEval/configs/VisDrone_test_dev.yaml

```

Similarly, you also need to modify the 'data_root' and 'trackers_folder' in the YAML file to specify your ground truth 
and tracking results folder.

For single category reviews, please ensure that each line of your tracking results follows the following format:
```
<frame id>,< object id>,<top-left-x>,<top-left-y>,<w>,<h>,<confidence score>,-1,-1,-1
```

For multi-category reviews, please ensure that each line of your tracking results follows the following format:
```
<frame id>,< object id>,<top-left-x>,<top-left-y>,<w>,<h>,<confidence score>,<class_id>,<class_id>,<class_id>
```

Note that in the process of multi-class evaluation, the 'class_id' in your tracking results must be completely 
consistent with the * * marked as true value * *. For example, for VisDrone, the valid category IDs are '1, 4, 5, 6, 9' 
(corresponding to pedestrian, car, van, truck, bus), so the class ID part of your tracking result must also correspond 
to '1, 4, 5, 6, 9', instead of '0, 3, 4, 5, 8' directly obtained by the detector. This requires you to modify the part 
of the tracking code that writes the tracking results yourself.

* **Evaluating on UAVDT**

The annotation of UAVDT dataset is divided into three files` gt.txt, gt_whole.txt, gt_ignore.txt `. Among them, 
` gt.txt ` is the main annotation, while ` gt_ignore.txt ` is the annotation of the area that should be ignored. 
Therefore, we should merge these two files to **exclude matches within the ignored area, otherwise it will create an 
oversized false positive (FP)** function. You run this only once.


```bash
# Being in 'DetectTrack-UAV/' folder
python Easier_To_Use_TrackEval/dataset_tools/parse_uavdt_annotations.py --data_root ./trackingdatasets/UAVDT/
```

Subsequently run

```bash
# Being in 'DetectTrack-UAV/' folder
python Easier_To_Use_TrackEval/scripts/run_custom_dataset.py --config_path Easier_To_Use_TrackEval/configs/UAVDT_test.yaml
```

Similarly, you also need to modify the 'data_root' and 'trackers_folder' in the YAML file to specify your ground truth 
and tracking results folder. Be careful, UAVDT has a total of 50 videos, of which 20 videos are from the test set, 
which is in UAVDT_test.yaml

Please ensure that each line of your tracking results follows the following format:
```
<frame id>,< object id>,<top-left-x>,<top-left-y>,<w>,<h>,<confidence score>,-1,-1,-1
```


### Tracking Demo
DetectTrack-UAV demo based on fine-tuned YOLOX-X detector for three different demo types: webcam, image and video.

```shell
# Using webcam, Being in 'DetectTrack-UAV/' folder 
python tools/demo_tracker_uav.py --demo_type webcam --camid 0 --experiment-name FSORTuav1 --fp16 --fuse --display_tracks --save_result

# Using image
python tools/demo_tracker_uav.py --demo_type image --path <path_to_images> --experiment-name FSORTuav1 --fp16 --fuse --display_tracks --save_result

# Using video
python tools/demo_tracker_uav.py --demo_type video --path <path_to_video> --experiment-name FSORTuav1 --fp16 --fuse --display_tracks --save_result
```


## Citation

If you find this work useful in your research, please consider citing it using:

```
@misc{DetectTrackUAV2025,
    title={{DetectTrack-UAV}: Multi-object Detection and Tracking in Moving UAV Videos},
    author={Nathanael L. Baisa},
    howpublished = {\url{https://github.com/nathanlem1/DetectTrack-UAV}},
    year={2025}
}
```


## Acknowledgement

We implemented our tracker based on top of publicly available codes. Hence, we would like to thank the authors of
[FusionSORT](https://github.com/nathanlem1/FusionSORT) and
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for making their code publicly available.
