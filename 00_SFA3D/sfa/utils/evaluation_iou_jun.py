from data_process.kitti_bev_utils import get_corners
import sys
import numpy as np

def rotMatrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def bev_corner(x,y,w,l,yaw):
    x1 = x - w/2 - l/2
    y1 = y - w/2 - l/2
    x2 = x + w/2 + l/2
    y2 = y + w/2 + l/2

    bevR = rotMatrix(yaw)
    
    # x1,y1 = np.dot(bevR,np.array([x1,y1]).T).T
    # x2,y2 = np.dot(bevR,np.array([x2,y2]).T).T
    points = [int(x1),int(y1),int(x2),int(y2)]
    print(points)

    return points
# #input : [x,y,w,l,yaw]
def get_points(xywly):
    x = xywly[0]
    y = xywly[1]
    w = xywly[2]
    l = xywly[3]
    yaw = xywly[4]
    
    bev_corners = bev_corner(x, y, w, l, yaw)
    # points = [bev_corners[0, 0], bev_corners[0, 1],bev_corners[3, 0],bev_corners[3, 1]]
    
    #[x1,y1,x2,y2]
    return bev_corners

# input : [x,y,w,l,yaw]
def calculate_iou(b1, b2):
    
    box1 = get_points(b1)
    box2 = get_points(b2)
    # print(box2)
    # print(box1)
    # box : (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return box1,box2 , iou