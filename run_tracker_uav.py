import argparse
import os
import sys
import os.path as osp
import cv2
import numpy as np
import torch

sys.path.append('.')

from loguru import logger

from tracker.fusion_sort_uav import FusionSORTUAV
from tracker.tracking_utils.timer import Timer
from tracker.tracking_utils.visualization import plot_tracking

from yoloxdetector.yolox.data.data_augment import preproc
from yoloxdetector.yolox.exp import get_exp
from yoloxdetector.yolox.utils import fuse_model, get_model_info, postprocess

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# Global
trackerTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser(description='Tracking pipeline for tracking objects in UAV videos.')

    parser.add_argument("--path", default='./trackingdatasets/VisDrone2019/MOT/VisDrone2019-MOT-test-dev/sequences', type=str,
                        help="path to dataset under evaluation, currently only support VisDrone2019 and UAVDT."
                             "./trackingdatasets/VisDrone2019/MOT/VisDrone2019-MOT-test-dev/sequences  OR "  
                             "./trackingdatasets/UAVDT/UAV-benchmark-M")
    parser.add_argument('--output_dir', type=str, default='./results/trackers',
                        help='Path to base tracking result folder to be saved to.')
    parser.add_argument("--benchmark", dest="benchmark", type=str, default='VisDrone',
                        help="benchmark to evaluate: VisDrone | UAVDT (VisDrone2019 and UAVDT datasets).")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='test',
                        help="split to evaluate: train | val | test (currently only test is supported - just for "
                             "evaluation!)")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default='FSORTuav3',
                        help='The name of the experiment, used for running different experiments and then evaluations, '
                             'e.g. FSORTuav1, FSORTuav2, etc.')
    parser.add_argument("--default-parameters", dest="default_parameters", default=True, action="store_true",
                        help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=True, action="store_true",
                        help="save sequences with tracks.")
    parser.add_argument('--display_tracks', default=False, action="store_true", help='Display sequences with tracks.')

    # Detector
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",
                        help="Adopting mix precision evaluation.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6,
                        help="tracking confidence threshold for the first association")
    parser.add_argument("--track_low_thresh", default=0.1, type=float,
                        help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument('--with_nsa', default=False, action='store_true',
                        help='For using Noise Scale Adaptive (NSA) Kalman Filter (R_nsa = (1-detection score)R')

    # CMC
    parser.add_argument("--cmc-method", default="none", type=str,
                        help="cmc method: sparseOptFlow | orb | sift | ecc | none")

    # Weak cues
    parser.add_argument('--iou_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--with-hiou', dest='with_hiou', default=False, action='store_true',
                        help='For using weak clue, particularly height-IoU distance.')
    parser.add_argument('--with-confidence', dest='with_confidence', default=False, action='store_true',
                        help='For using weak clue, particularly confidence distance.')

    # Fusion (iou with weak cues)
    parser.add_argument('--lambda1', type=float, default=0.1,
                        help='Value for lambda 1 - weight for height-IoU (for weighted_sum')
    parser.add_argument('--lambda2', type=float, default=0.1,
                        help='Value for lambda 2 - weight for tracklet confidence (for weighted_sum')

    parser.add_argument('--second_matching_distance', default='iou', type=str,
                        help='Matching distance for the second matching: iou or mahalanobis')

    # Tracker Evaluation
    parser.add_argument('--multi_class_eval', default=False, action='store_true', help='For multi-class evaluation.')

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        if img is None:
            raise ValueError("Empty image: ", img_info["file_name"])

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img, ratio = preproc(img, self.test_size)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

        return outputs, img_info


def image_track(predictor, vis_folder, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()

    num_frames = len(files)

    # Tracker
    tracker = FusionSORTUAV(args, frame_rate=args.fps)

    results = []

    for frame_id, img_path in enumerate(files, 1):

        # Detect objects
        outputs, img_info = predictor.inference(img_path, timer)
        scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale

            # Select classes to track
            if args.benchmark == 'VisDrone':  # Track five classes: pedestrian, car, van, truck and bus.
                detections_p = detections[np.where(detections[:, 6] == 0)]  # pedestrian
                detections_c = detections[np.where(detections[:, 6] == 3)]  # car
                detections_v = detections[np.where(detections[:, 6] == 4)]  # van
                detections_t = detections[np.where(detections[:, 6] == 5)]  # truck
                detections_b = detections[np.where(detections[:, 6] == 8)]  # bus
                detections = np.concatenate((detections_p, detections_c, detections_v, detections_t, detections_b), axis=0)
            else:  # Track only one class, car, for UAVDT dataset
                detections = detections[np.where(detections[:, 6] == 3)]

            trackerTimer.tic()
            online_targets = tracker.step(detections, img_info["raw_img"])
            trackerTimer.toc()

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)  # We use the corresponding detection saved score for display purpose.

                    # Save results
                    if args.benchmark == 'VisDrone' and args.multi_class_eval:
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},"
                            f"{int(t.category + 1):d}, {int(t.category + 1):d}, {int(t.category + 1):d}\n")  # 1 is
                        # added for evaluation purposes, read more on README.md.
                    else:
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},"
                            f"-1, -1, -1\n")
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        if args.save_frames:
            save_folder = osp.join(vis_folder, args.name)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        # Display tracks
        if args.display_tracks:
            cv2.imshow('Tracking', online_im)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        if frame_id % 20 == 0:
            logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time)))

    res_file = osp.join(vis_folder, args.name + ".txt")

    with open(res_file, 'w') as f:
        f.writelines(results)
    logger.info(f"save results to {res_file}")


def main(exp, args):

    vis_folder = os.path.join(args.output_dir, args.benchmark + '-' + args.split_to_eval, args.experiment_name, 'data')
    os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Detector Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if args.ckpt is None:
        output_dir = os.path.join(exp.output_dir, exp.exp_name)
        ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")

    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    predictor = Predictor(model, exp, args.device, args.fp16)

    image_track(predictor, vis_folder, args)


if __name__ == "__main__":
    args = make_parser().parse_args()

    data_path = args.path
    fp16 = args.fp16
    device = args.device

    if args.benchmark == 'VisDrone':
        train_seqs = []
        val_seqs = []
        # test_seqs = ['uav0000297_00000_v']
        test_seqs = ['uav0000009_03358_v', 'uav0000073_00600_v', 'uav0000073_04464_v', 'uav0000077_00720_v', 'uav0000088_00290_v',
                     'uav0000119_02301_v', 'uav0000120_04775_v', 'uav0000161_00000_v', 'uav0000188_00000_v', 'uav0000201_00000_v',
                     'uav0000249_00001_v', 'uav0000249_02688_v', 'uav0000297_00000_v', 'uav0000297_02761_v', 'uav0000306_00230_v',
                     'uav0000355_00001_v','uav0000370_00001_v']
        seqs_ext = ['']
        MOT = ''
    elif args.benchmark == 'UAVDT':
        train_seqs = []
        val_seqs = []
        test_seqs = ['M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606', 'M0701', 'M0801', 'M0802',
                     'M1001', 'M1004', 'M1007', 'M1009', 'M1101', 'M1301', 'M1302', 'M1303', 'M1401']
        seqs_ext = ['']
        MOT = ''
    else:
        raise ValueError("Error: Unsupported benchmark:" + args.benchmark)

    ablation = False
    if args.split_to_eval == 'train':
        seqs = train_seqs
        split = 'train'
    elif args.split_to_eval == 'val':
        seqs = train_seqs
        split = 'train'
        ablation = True
    elif args.split_to_eval == 'test':
        seqs = test_seqs
        split = 'test'
    else:
        raise ValueError("Error: Unsupported split to evaluate:" + args.split_to_eval)

    mainTimer = Timer()
    mainTimer.tic()

    for ext in seqs_ext:
        for i in seqs:
            if args.benchmark == 'VisDrone':
                seq = i
            elif args.benchmark == 'UAVDT':
                seq = i
            else:
                raise ValueError("Error: Unsupported benchmark:" + args.benchmark)

            if ext != '':
                seq += '-' + ext

            args.name = seq

            args.ablation = ablation
            args.mot20 = MOT == 20
            args.fps = 30
            args.device = device
            args.fp16 = fp16
            args.batch_size = 1
            args.trt = False

            if args.benchmark == 'VisDrone' or args.benchmark == 'UAVDT':
                args.path = data_path + '/' + seq

            if args.default_parameters:
                if args.benchmark == 'VisDrone':
                    args.exp_file = './yoloxdetector/exps/example/custom/yolox_x_weakaug_640.py'
                    args.ckpt = r'./yoloxdetector/pretrained/yolox_best_ckpt_640.pth'    # Detection, VisDrone
                    # args.exp_file = './yoloxdetector/exps/example/custom/yolox_x_weakaug_2048.py'
                    # args.ckpt = r'./yoloxdetector/pretrained/yolox_best_ckpt_2048.pth'    # Detection, VisDrone
                elif args.benchmark == 'UAVDT':
                    args.exp_file = './yoloxdetector/exps/example/custom/yolox_x_weakaug_640.py'
                    args.ckpt = r'./yoloxdetector/pretrained/yolox_best_ckpt_640.pth'    # Detection, VisDrone
                    # args.exp_file = './yoloxdetector/exps/example/custom/yolox_x_weakaug_2048.py'
                    # args.ckpt = r'./yoloxdetector/pretrained/yolox_best_ckpt_2048.pth'    # Detection, VisDrone
                else:
                    raise ValueError("Error: Unsupported benchmark:" + args.benchmark)

                exp = get_exp(args.exp_file, args.name)

                args.track_high_thresh = 0.6
                args.track_low_thresh = 0.1
                args.track_buffer = 30

                args.new_track_thresh = args.track_high_thresh + 0.1
            else:
                exp = get_exp(args.exp_file, args.name)

            exp.test_conf = max(0.001, args.track_low_thresh - 0.01)

            # Call main function
            main(exp, args)

    mainTimer.toc()
    print("TOTAL TIME END-to-END (with loading networks and images): ", mainTimer.total_time)
    print("TOTAL TIME (Detector + Tracker): " + str(timer.total_time) + ", FPS: " + str(1.0 / timer.average_time))
    print("TOTAL TIME (Tracker only): " + str(trackerTimer.total_time) + ", FPS: " + str(1.0 / trackerTimer.average_time))


