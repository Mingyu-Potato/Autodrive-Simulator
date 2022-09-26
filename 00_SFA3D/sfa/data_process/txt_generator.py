import os

# pcd, json이 둘 다 있을 경우에만 test.txt에 입력
lidar_list = os.listdir("/home/krri/nas_student/김민규/IOU_eval/testing/velodyne")
lidar_list.sort()

label_list = os.listdir("/home/krri/nas_student/김민규/IOU_eval/testing/label_2")
label_list.sort()

for lidar_file in lidar_list:
    file_name = lidar_file.split('.')[0]
    if file_name + ".json" in label_list:
        with open("/home/krri/nas_student/김민규/IOU_eval/ImageSets/test.txt", "a") as f:
            f.write(file_name + "\n")