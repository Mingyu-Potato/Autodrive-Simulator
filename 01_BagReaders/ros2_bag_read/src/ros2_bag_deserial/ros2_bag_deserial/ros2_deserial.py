from pyrsistent import field
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import rosbag2_py
import open3d as o3d
from . import pointcloud2 as rnp
import os
import numpy as np
import cv2
import pandas as pd
import yaml
from cv_bridge import CvBridge

device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32

set_file_path = './src/ros2_bag_deserial/bag_setting/setting.yaml'

class ReadBag():
    def __init__(self):
        global output_path
        bag_set = self.load_setting()
        self.save_dir = ''
        self.bag_uri = bag_set['bag_uri']
        output_path = bag_set['output_path']
        self.topic_list = bag_set['lidar_topics'] + bag_set['cam_topics'] + bag_set['custom_topics'] + bag_set['csv_topics']
        self.lidar_topics = bag_set['lidar_topics']
        self.lidar_color = bag_set['lidar_color']
        self.cam_topics = bag_set['cam_topics']
        self.csv_topics = bag_set['csv_topics']
        self.custom_topics = bag_set['custom_topics']
        self.option_list = bag_set['option_list']
        self.cam_sim_frame = '1' # ObsData sim_frame 기준
        # self.all_option = bag_set['all_option']
        self.sim_data = bag_set['sim_data'] #save sim_frame
        
    def load_setting(self):
        with open(set_file_path, 'r', encoding='utf-8') as f:
            bag_set = yaml.load(f, Loader=yaml.FullLoader)
            return bag_set

    # lidar bag to pcd and save
    def bag_to_pcd(self, lidar_msg, t, dir_name):
        file_name = self.set_name(t, dir_name)
        file_name = file_name + ".pcd"
        
        msg_np = rnp.pointcloud2_to_array(lidar_msg)
        
        xyz_name = ['x','y','z']
        field_names = [f.name for f in lidar_msg.fields if f.name not in xyz_name]

        msg_xyz = []
        for xyz in xyz_name:
            msg_xyz.append(msg_np[xyz].reshape(-1,1))
        
        pcd_xyz = np.concatenate(msg_xyz, axis=1)
        
        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point['positions'] = o3d.core.Tensor(pcd_xyz, dtype, device)
        # pcd.point["intensities"] = o3d.core.Tensor(msg_np['intensity'].reshape(-1,1), dtype, device)
        if self.lidar_color:
            rgb_name = ['r','g','b']
            msg_rgb = []
            for rgb in rgb_name:
                msg_rgb.append(msg_np[rgb].reshape(-1,1))
            pcd_rgb = np.concatenate(msg_rgb, axis=1)
            pcd.point['colors'] = o3d.core.Tensor(pcd_rgb, dtype, device)
            
        # if field name is rgb, you get strange pcd data
        # so please change other name
        field_names = field_names[1:]

        for f in np.flip(field_names):
            pcd.point[f] = o3d.core.Tensor(msg_np[f].reshape(-1,1), dtype, device)
            

        o3d.t.io.write_point_cloud(file_name, pcd)
    
    # bag to img // 그냥 이미지
    # bag to img and save // 압축이미지
    def bag_to_img(self, cam_msg, t, dir_name):
        
        #---------------- raw image -----------------------
        # br = CvBridge()
        # img = br.imgmsg_to_cv2(cam_msg,'bgr8')
        
        #-----------------Compressed-----------------------
        np_img = np.frombuffer(cam_msg.data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        file_name = self.set_name(t, dir_name)
        cv2.imwrite(file_name+'.jpg', img)

    # msg data to csv and save    
    def msg_to_csv(self, msg, t, dir_name):
        attr_list = []
        for msg_attr in dir(msg):
            for option in self.option_list:
                if(msg_attr == option):
                    attr_list.append(msg_attr)
        msg_data = []
        for attr in attr_list:
            msg_data.append(getattr(msg,attr))
            if(attr == 'sim_frame'): t = str(getattr(msg, attr))
        msg_data = pd.DataFrame(msg_data, index=attr_list)
        file_name = self.set_name(t, dir_name)
        msg_data.to_csv(file_name + '.csv', mode='w')
    
    '''
    obj_to_csv : custom msg topic function
    point_to_array : use obj get x, y, z points
    obj to csv and save
    '''
    '''custom function'''
    def obj_to_csv(self, obs_msg, t, dir_name):
        self.cam_sim_frame = str(obs_msg.sim_frame)
        objs = obs_msg.obj
        attrs = [att for att in dir(objs[0]) if att.split('_')[0] == 'front' or att.split('_')[0] == 'rear']
        obj_arr = []
        index_arr = []
        for index, obj in enumerate(objs):
            for attr in attrs:
                xyz = self.point_to_array(getattr(obj, attr))
                obj_arr.append(xyz)
                index_arr.append(index)
        obj_df = pd.DataFrame(obj_arr, index=index_arr, columns=['x','y','z'])
        
        if(self.sim_data):
            file_name = self.set_name(self.cam_sim_frame, dir_name)
        else:
            file_name = self.set_name(t, dir_name)
        
        obj_df.to_csv(file_name +'.csv', mode='w')
    
    def point_to_array(self, point):
        xyz = np.array([point.x, point.y, point.z])
        return xyz
    '''custom function'''
    
    
    # set file name before save
    def set_name(self, t, dir_name):
        if(self.sim_data):
            file_name = self.save_dir + dir_name + '/' + str(t)
        else:
            file_name = self.save_dir + dir_name + '/' + str(int(t/1000000))
        return file_name

    # read bag
    def run(self):
        options = rosbag2_py.StorageOptions('','')
        options.uri = self.bag_uri
        options.storage_id = 'sqlite3'

        converter = rosbag2_py.ConverterOptions('','')
        converter.input_serialization_format = 'cdr'
        converter.output_serialization_format = 'cdr'

        reader = rosbag2_py.SequentialReader()
        
        try:
            reader.open(options, converter)
        except Exception as e:
            print("error")

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic_types[i].name : topic_types[i].type for i in range(len(topic_types))}

        storage_filter = rosbag2_py.StorageFilter(self.topic_list)
        reader.set_filter(storage_filter)

        i = 0
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            
            if(topic in self.lidar_topics):
                sim_frame = msg.header.stamp.sec
                
                if(self.sim_data):
                    timestamp = sim_frame 

                self.bag_to_pcd(msg, timestamp, topic)
            
            if(topic in self.cam_topics):
                if(self.sim_data):
                    timestamp = self.cam_sim_frame
                    
                self.bag_to_img(msg, timestamp, topic)
            
            # ObsData
            if(topic in self.custom_topics):
                self.obj_to_csv(msg, timestamp, topic)
            
            # ego_pose
            if(topic in self.csv_topics):
                if(self.sim_data):
                    timestamp = self.cam_sim_frame
                
                self.msg_to_csv(msg, timestamp, topic)
            
            i += 1
            print(f'processing..{i}\r', end='')

        print("complete")

# main func make dir before read bag
def main():
    readbag = ReadBag()
    bag_name = readbag.bag_uri.split('/')[-1]
    bag_name = bag_name.split('.')[0]
    
    data_dir = output_path + "/" + bag_name
    readbag.save_dir = data_dir
    if (not os.path.isdir(data_dir)):
        os.makedirs(data_dir)
    
    for topic_name in readbag.topic_list:
        if(not os.path.isdir(data_dir + topic_name)):
            os.makedirs(data_dir + topic_name)
    
    readbag.run()
