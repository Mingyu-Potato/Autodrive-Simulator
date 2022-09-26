import os
import numpy as np
import cv2
import open3d as o3d
from PIL import Image, ImageGrab
import pyscreenshot

from pcd2bevmap import *
import config as cnf


def pcd2bev(src_path, pcd_file):
    lidarData = get_lidar(src_path, pcd_file)
    lidarData = get_filtered_lidar(lidarData, cnf.boundary)
    bev_map = makeBEVMap(lidarData, cnf.boundary)
    bev_map = bev_map.transpose((1,2,0))
    
    return bev_map


if __name__ == '__main__':
    # 모든 파일들을 dictionary에 저장
    file_list_dict = dict()
    for i in range(1, 9):
        if i <= 4:
            file_path = f'./sync/c{str(i)}'
            file_list = os.listdir(file_path)
            file_list.sort()

            file_list_dict[f'c{str(i)}'] = file_list

        else:
            file_path = f'./sync/l{str(i-4)}'
            file_list = os.listdir(file_path)
            file_list.sort()

            file_list_dict[f'l{str(i-4)}'] = file_list

    # c1~c4, l1~l4 파일들을 2x4로 concat
    dst_path = './sync/sync_check'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for idx, _ in enumerate(range(len(file_list_dict['c1']))):
        c1 = cv2.imread(os.path.join('./sync/c1', file_list_dict['c1'][idx]))
        c2 = cv2.imread(os.path.join('./sync/c2', file_list_dict['c2'][idx]))
        c3 = cv2.imread(os.path.join('./sync/c3', file_list_dict['c3'][idx]))
        c4 = cv2.imread(os.path.join('./sync/c4', file_list_dict['c4'][idx]))
        l1 = pcd2bev('./sync/l1', file_list_dict['l1'][idx])
        l2 = pcd2bev('./sync/l2', file_list_dict['l2'][idx])
        l3 = pcd2bev('./sync/l3', file_list_dict['l3'][idx])
        l4 = pcd2bev('./sync/l4', file_list_dict['l4'][idx])

        c1 = cv2.resize(c1, dsize=(0, 0), fx=0.43, fy=0.43, interpolation=cv2.INTER_LINEAR)
        c2 = cv2.resize(c2, dsize=(0, 0), fx=0.43, fy=0.43, interpolation=cv2.INTER_LINEAR)
        c3 = cv2.resize(c3, dsize=(0, 0), fx=0.43, fy=0.43, interpolation=cv2.INTER_LINEAR)
        c4 = cv2.resize(c4, dsize=(0, 0), fx=0.43, fy=0.43, interpolation=cv2.INTER_LINEAR)
        l2 = cv2.resize(l2, dsize=(0, 0), fx=0.51, fy=0.51, interpolation=cv2.INTER_LINEAR)
        l1 = cv2.resize(l1, dsize=(0, 0), fx=0.51, fy=0.51, interpolation=cv2.INTER_LINEAR)
        l3 = cv2.resize(l3, dsize=(0, 0), fx=0.51, fy=0.51, interpolation=cv2.INTER_LINEAR)
        l4 = cv2.resize(l4, dsize=(0, 0), fx=0.51, fy=0.51, interpolation=cv2.INTER_LINEAR)

        l1 = cv2.rotate(l1, cv2.ROTATE_180)
        l2 = cv2.rotate(l2, cv2.ROTATE_180)
        l3 = cv2.rotate(l3, cv2.ROTATE_180)
        l4 = cv2.rotate(l4, cv2.ROTATE_180)


        ch1 = np.hstack((c1,c2))
        ch2 = np.hstack((c3,c4))
        cam_stack = np.vstack((ch1, ch2))

        lh1 = np.hstack((l1,l2))
        lh2 = np.hstack((l3,l4))
        lidar_stack = np.vstack((lh1, lh2))

        cv2.namedWindow('cam')
        cv2.namedWindow('lid')
        cv2.moveWindow('cam', 10, 10)
        cv2.moveWindow('lid', 1180, 10)
        cv2.imshow('cam', cam_stack)
        cv2.imshow('lid', lidar_stack)
        cv2.waitKey(100)
        image = pyscreenshot.grab(bbox=(60,30,1810,740))
        # image.show()
        image.save(f"./sync/concat_2/{str(idx+1).zfill(6)}.jpg")

        # cv2.waitKey(0)
        cv2.destroyAllWindows()