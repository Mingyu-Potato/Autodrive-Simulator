from csv import list_dialects
from rosbag import Bag
from sensor_msgs.msg import *
from pointcloud2 import pointcloud2_to_array  as rnp
import open3d as o3d
import cv2
import os
import numpy as np
import yaml
from datetime import datetime
from numba import jit

device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32

""" 대용량 BAG 데이터 파일 추출 하는 방법
1. 대용량 bag 파일을 ssd 로 가져온다
2. 터미널창 킨 뒤 home/krri/catkin_ws 폴더에서 . devel/setup.bash
3. rosrun rosbag rosbag index-file 2020-11-12-11-31-32_1.bag 하면 해당 bag 이름과 같은 idx 파일 생성됨
    (idx 파일과 bag 파일은 같은 폴더에 놓고 bag 접근 시 로딩 시간 없이 bag 을 열수있다.)
4.  idx 파일 생성 후 python 코드 내에서 output path 를 NAS 폴더로 지정 해서 추출하는 방법이 가장 효율적임

 # 3.5GB Bag 파일로 로 테스트 한 결과
 코드 실행위치 -> 추출 결과 저장 위치 : 걸린 시간 
 HDD -> HDD  : 1m 12s
 SSD -> SSD  : 18s
 NAS -> NAS  : 1m 2s
 SSD -> NAS  : 20s

코드 Config 설정
bag_uri : bag이 저장된 위치 (ssd 에 위치할수록 처리속도가 빠르며 idx 파일이 함께 있을경우 초기 bag 로딩 속도가 빠르다)
output_dir_name : 저장할 폴더 명
output_path : 저장할 위치 
camera_topics : 이미지 데이터롤 뽑을 카메라 Topic (raw 이미지만 가능)
lidar_topics : pcd 로 뽑을 라이다 Topic

@ 더 많은 라이다 값을 뽑고싶은 경우 >> bag_to_pcd 함수 내에서
    pcd.point["t"] = o3d.core.Tensor(msg_np['t'].reshape(-1,1), dtype, device)
    과 같이 ["t"] 위치에 라이다가 가지고있는 field 명을 입력하여 한줄 추가하면 됨
"""

class ReadBag():
    def __init__(self):
        global output_path
        self.bag_uri = '/home/krri/SSD/2020-11_bags/2020-11-12-11-33-13.bag'
        output_dir_name = '/raw'
        output_path = '/home/krri/brtdata/exp_20201111' + output_dir_name #os.path.abspath(os.path.join(self.bag_uri, '..')) + output_dir_name
        self.bag = Bag(self.bag_uri)
        self.camera_topics = ['/camera1/usb_cam1/image_raw','/camera2/usb_cam2/image_raw','/camera3/usb_cam3/image_raw','/camera4/usb_cam4/image_raw']
        self.lidar_topics = ['/os1_cloud_node/points']
        self.topic_list = self.camera_topics + self.lidar_topics
        self.save_dir = ''
        
    # bag to raw img
    def bag_to_img(self, cam_msg, t, dir_name):
        np_img = np.frombuffer(cam_msg.data, np.uint8)
        img = np.reshape(np_img,(cam_msg.height,cam_msg.width,3))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        file_name = self.set_name(t, dir_name)
        cv2.imwrite(file_name+'.jpg', img)

    def bag_to_pcd(self, lidar_msg, t, dir_name):
        file_name = self.set_name(t, dir_name)
        file_name = ''.join([file_name, ".pcd"])
        
        msg_np = rnp(lidar_msg)
        
        xyz_name = ['x','y','z']
        field_names = [f.name for f in lidar_msg.fields if f.name not in xyz_name]

        msg_xyz = []
        msg_xyz = [msg_np[xyz].reshape(-1,1) for xyz in xyz_name]
        
        pcd_xyz = np.concatenate(msg_xyz, axis=1)
        
        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point['positions'] = o3d.core.Tensor(pcd_xyz, dtype, device)
        
        pcd.point["range"] = o3d.core.Tensor(msg_np['range'].reshape(-1,1), dtype, device)
        pcd.point["ring"] = o3d.core.Tensor(msg_np['ring'].reshape(-1,1), dtype, device)
        # pcd.point["noise"] = o3d.core.Tensor(msg_np['noise'].reshape(-1,1), dtype, device)
        pcd.point["t"] = o3d.core.Tensor(msg_np['t'].reshape(-1,1), dtype, device)
        pcd.point["intensities"] = o3d.core.Tensor(msg_np['intensity'].reshape(-1,1), dtype, device)
        
        # if field name is rgb, you get strange pcd data
        # so please change other name
        # field_names = field_names[1:]
        # print(field_names)
        # for f in np.flip(field_names):
        #     print(f)
        #     pcd.point[f] = o3d.core.Tensor(msg_np[f].reshape(-1,1), dtype, device)
        
        o3d.t.io.write_point_cloud(file_name, pcd)

    def set_name(self, t, dir_name):
        file_name = ''.join([self.save_dir, dir_name, '/', str(t)])
        return file_name

    def run(self):
        print("start!")
        i = 0
        
        for topic, msg, t in self.bag.read_messages(topics = self.topic_list):
            start = datetime.now()
            if (topic in self.lidar_topics):
                self.bag_to_pcd(msg,t,topic)
            
            if (topic in self.camera_topics):
                img_start = datetime.now()
                self.bag_to_img(msg,t,topic)
                img_end = datetime.now()
                # time = f'img time..{img_end-img_start}\r'
                # print(time)
            i += 1
            end = datetime.now()
            print(f'1 for time..{end-start}',end='\r')
            if (i%10 == 0):
                print(f'processing..{i}', end='\r')
            
            
        self.bag.close()
        

print("Loading Bag....")

readbag = ReadBag()
bag_name = readbag.bag_uri.split('/')[-1]
bag_name = bag_name.split('.')[0]

data_dir = output_path +"/"+ bag_name
readbag.save_dir = data_dir
if (not os.path.isdir(data_dir)):
    os.makedirs(data_dir)

for topic_name in readbag.topic_list:
    if(not os.path.isdir(data_dir + topic_name)):
        os.makedirs(data_dir + topic_name)
        
start_time = datetime.now()
print("Starting time >> {0} ".format(start_time))
readbag.run()
end_time = datetime.now()
diff = end_time - start_time
print("Finished >> {0}".format(end_time))
print("Time >> {0}".format(diff))
