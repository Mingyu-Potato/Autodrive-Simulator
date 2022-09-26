"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

import argparse
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np
import shutil

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_dataloader import create_test_dataloader
from data_process.kitti_dataset import drawRotatedBox, KittiDataset
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, draw_compared_image, convert_det_to_real_values
from utils.torch_utils import _sigmoid
from utils.evaluation_iou import *
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='/home/krri/Desktop/SFA3D/checkpoints/fpn_resnet_18/pretrained/Model_fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 5 # 수정
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    # /home/krri/brtdata/철도_샘플데이터/2021_1013_day에서 3d_label 폴더는 학습할때 사용, lidar는 그냥 사용(이 경로는 sample data의 경로)
    # /home/krri/brtdata/2021_06_BAG/raw/Train_Test Dataset/Train/Day/D1 (실제                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         데이터로써, 라벨링 업체에 보낸 data)
    configs.dataset_dir = "/home/krri/nas_student/김민규/IOU_eval"
    # configs.dataset_dir = "/home/krri/nas_student/김민규/SFA3D_ERROR_BOX/detection_only_box"
    # configs.dataset_dir = "/home/krri/nas_student/김민규/SFA3D_ERROR_BOX/GT_only_box"
    # configs.dataset_dir = "/home/krri/nas_student/김민규/SFA3D_ERROR_BOX/class_error_box"

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    start_time = time.time()

    configs = parse_test_configs()

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    test_dataloader = create_test_dataloader(configs)
    with torch.no_grad():
        total_box = 0
        iou_over_50_box_strict = 0
        iou_over_50_box = 0
        class_error_box = 0
        detection_only_box_0 = 0
        detection_only_box_0_over_50_under = 0
        gt_only_box_0 = 0
        gt_only_box_0_over_50_under = 0

        for batch_idx, batch_data in enumerate(test_dataloader):
            bev_maps, sample_id = batch_data
            input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
            t1 = time_synchronized()
            outputs = model(input_bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            # detections size (batch_size, K, 10)
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            t2 = time_synchronized()

            detections = detections[0]  # only first batch

            # # detection 결과에서 같은 물체에 박스가 두 개 이상 쳐진 경우(iou가 5% 이상인 경우), score가 높은 것을 택하고 나머지는 제거(같은 클래스 일 경우에만)
            # for j in range(configs.num_classes):
            #     if len(detections[j]) >= 2:
            #         detections[j] = remove_overlap_box(configs, detections[j], threshold=0.05)

            # detection 결과에서 같은 물체에 박스가 두 개 이상 쳐진 경우(iou가 5% 이상인 경우), score가 높은 것을 택하고 나머지는 제거(같은 클래스 일 경우에만)
            detections = remove_overlap_box2(configs, detections.copy(), threshold=0.05)

            # Draw prediction in the image
            detection = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            detection = cv2.resize(detection, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            detection = draw_predictions(detection, detections.copy(), configs.num_classes)

            # yolo_detection result
            # -------------------------------------------------
            # Ground Truth 
            lidar_aug = None
            dataset = KittiDataset(configs, mode='test', lidar_aug=lidar_aug, hflip_prob=0., num_samples=configs.num_samples)

            ground_truth, labels, sample_id = dataset.draw_img_with_label(batch_idx)
            # calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
            ground_truth = (ground_truth.transpose(1, 2, 0) * 255).astype(np.uint8)
            ground_truth = cv2.resize(ground_truth, (cnf.BEV_HEIGHT, cnf.BEV_WIDTH))

            for box_idx, (cls_id, x, y, z, h, w, l, yaw) in enumerate(labels):
                # Draw rotated box
                yaw = -yaw
                y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
                x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
                w1 = int(w / cnf.DISCRETIZATION)
                l1 = int(l / cnf.DISCRETIZATION)

                drawRotatedBox(ground_truth, x1, y1, w1, l1, yaw, cnf.colors[int(cls_id)])
            # -------------------------------------------------
            # compared_image
            img = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = cv2.resize(img, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            compared_image = draw_compared_image(configs, img, detections.copy(), labels)

            # -----------------------------------------------------------------------
            # TP, FN 계산
            total_box += len(labels) # GT의 box 개수를 더한다.

            for box_idx, (cls_id, x, y, z, h, w, l, yaw) in enumerate(labels): # GT box 1개마다 모든 Detection box와 iou를 계산한다.
                gt_class_name = cnf.idx_2_class[int(cls_id)]
                yaw = -yaw
                y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
                x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
                w1 = int(w / cnf.DISCRETIZATION)
                l1 = int(l / cnf.DISCRETIZATION)

                ground_truth_input = [x1, y1, w1, l1, yaw]

                max_gt_iou = 0
                for j in range(configs.num_classes):
                    if len(detections[j]) > 0:
                        for det in detections[j]:
                            detect_class_name = cnf.idx_2_class[j]
                            _score, _x, _y, _z, _h, _w, _l, _yaw = det

                            detection_input = [_x, _y, _w, _l, _yaw]

                            gt_iou = calculate_iou(configs, ground_truth_input, detection_input)

                            if gt_iou >= 0.5: # iou가 0.5 이상이면, TP += 1, max_iou = iou
                                max_gt_iou = gt_iou
                                if detect_class_name == gt_class_name:
                                    iou_over_50_box_strict += 1
                                    iou_over_50_box += 1
                                elif detect_class_name in ['car', 'bus', 'truck'] and gt_class_name in ['car', 'bus', 'truck']:
                                    iou_over_50_box += 1
                                else:
                                    # shutil.copyfile(f'/home/krri/nas_student/김민규/IOU_eval/testing/velodyne/{sample_id}.pcd', f'/home/krri/nas_student/김민규/SFA3D_ERROR_BOX/class_error_box/testing/velodyne/{sample_id}.pcd')
                                    class_error_box += 1
                            elif (gt_iou > 0 ) and (gt_iou < 0.5): # iou가 0초과 0.5 미만이면, gt_only_box_0_over_50_under += 1, max_iou = iou
                                max_gt_iou = gt_iou
                                gt_only_box_0_over_50_under += 1
                            
                if max_gt_iou == 0:
                    gt_only_box_0 += 1

            # -------------------------------------------------
            # FP 계산
            for j in range(configs.num_classes):
                if len(detections[j]) > 0:
                    for det in detections[j]:
                        _score, _x, _y, _z, _h, _w, _l, _yaw = det
                        # detect_class_name = cnf.idx_2_class[j]

                        detection_input = [_x, _y, _w, _l, _yaw]
                        max_det_iou = 0

                        for box_idx, (cls_id, x, y, z, h, w, l, yaw) in enumerate(labels):
                            # gt_class_name_ = cnf.idx_2_class[int(cls_id)]
                            yaw = -yaw
                            y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
                            x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
                            w1 = int(w / cnf.DISCRETIZATION)
                            l1 = int(l / cnf.DISCRETIZATION)

                            ground_truth_input = [x1, y1, w1, l1, yaw]

                            det_iou = calculate_iou(configs, detection_input, ground_truth_input)

                            if det_iou >= 0.5:
                                max_det_iou = det_iou
                            elif (det_iou > 0) and (det_iou < 0.5):
                                max_det_iou = det_iou
                                detection_only_box_0_over_50_under += 1
                            
                        if max_det_iou == 0:
                            detection_only_box_0 += 1

            # img_path = metadatas['img_path'][0]
            # img_rgb = img_rgbs[0].numpy()
            # img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
            # img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            # calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
            # kitti_dets = convert_det_to_real_values(detections)
            # if len(kitti_dets) > 0:
            #     kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
            #     img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)

            # out_img = merge_rgb_to_bev(img_bgr, bev_map, output_width=configs.output_width)
            # out_img = detection

            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,
                                                                                           1 / (t2 - t1)))
            # if configs.save_test_output:
            #     if configs.output_format == 'image':
            #         img_fn = os.path.basename(metadatas['img_path'][0])[:-4]
            #         cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), out_img)
            #     elif configs.output_format == 'video':
            #         if out_cap is None:
            #             out_cap_h, out_cap_w = out_img.shape[:2]
            #             fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            #             out_cap = cv2.VideoWriter(
            #                 os.path.join(configs.results_dir, '{}.avi'.format(configs.output_video_fn)),
            #                 fourcc, 30, (out_cap_w, out_cap_h))

            #         out_cap.write(out_img)
            #     else:
            #         raise TypeError

            ground_truth = cv2.rotate(ground_truth, cv2.ROTATE_180)
            detection = cv2.rotate(detection, cv2.ROTATE_180)
            compared_image = cv2.rotate(compared_image, cv2.ROTATE_180)
            
            out_img = np.concatenate((ground_truth, detection, compared_image), axis=1)

            # cv2.imshow('IOU_eval', out_img)
            # print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
            # if cv2.waitKey(0) & 0xFF == 27:
            #     break

            # cv2.imwrite(f'/home/krri/nas_student/김민규/SFA3D_ERROR_BOX/GT_only_box/outputs/{sample_id}.jpg', out_img)
    
    print('\n---------------------------------------------------------------------------')
    print('-- Detected Box Result --')
    print('TP')
    print(f'{iou_over_50_box_strict}(IOU_over_50_box_strict) / {total_box}(Total_box) = {iou_over_50_box_strict/total_box}')
    print(f'{iou_over_50_box}(IOU_over_50_box) / {total_box}(Total_box) = {iou_over_50_box/total_box}')
    print(f'{class_error_box}(class_error_box) / {total_box}(Total_box) = {class_error_box/total_box}')
    print('\nFN')
    print(f'{gt_only_box_0}(gt_only_box_0) / {total_box}(Total_box) = {gt_only_box_0/total_box}')
    print(f'{gt_only_box_0_over_50_under}(gt_only_box_0_over_50_under) / {total_box}(Total_box) = {gt_only_box_0_over_50_under/total_box}')
    print('\nFP')
    print(f'{detection_only_box_0}(detection_only_box_0) / {total_box}(Total_box) = {detection_only_box_0/total_box}')
    print(f'{detection_only_box_0_over_50_under}(detection_only_box_0_over_50_under) / {total_box}(Total_box) = {detection_only_box_0_over_50_under/total_box}')
    print('---------------------------------------------------------------------------')
    
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    exec_time = end_time - start_time
    print(f"\nExec Time : {int(exec_time // 60)}min {int(exec_time % 60)}sec")
