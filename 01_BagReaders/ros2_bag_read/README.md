# ros2_bag_read
- ros2 bag을 이용하여 녹화한 db3 파일에서 원하는 데이터를 추출하는 코드
- 현재는 Lidar, cam(compressed), csv 데이터만 추출가능
- Lidar는 pcd, Cam은 이미지, 나머지는 csv로 저장
- ros2 deserial의 참고 및 자체 프로젝트에 활용하기위해 제작된 코드이므로 다른 프로젝트에 적용이 어려울 수 있음
---

# 요구사항
- open3d >= 0.14.1
- **pip3**로 open3d를 설치하여야 0.13 이상의 버전이 설치 가능함
---

## 실행방법

- bag_setting 폴더의 setting파일에 원하는 토픽 이름을 입력 후 저장
- lidar, cam, custom topic을 제외한 나머지는 csv에 입력
- custom topic의 경우 ros2_deserial.py에 custom 토픽의 msg_type에 맞는 코드가 필요함

<code>
    ros2 run ros2_bag_deserial deserial
</code>
