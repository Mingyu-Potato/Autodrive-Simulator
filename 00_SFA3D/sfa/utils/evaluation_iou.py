from data_process.kitti_bev_utils import get_corners
import sys
import numpy as np
import cv2

# #input : [x,y,w,l,yaw]
def get_points(xywly):
    x = xywly[0]
    y = xywly[1]
    w = xywly[2]
    l = xywly[3]
    yaw = xywly[4]
    
    bev_corners = get_corners(x, y, w, l, yaw)
    
    # points = [bev_corners[1, 0], bev_corners[1, 1],bev_corners[3, 0],bev_corners[3, 1]]
    points = np.array([[bev_corners[0, 0], bev_corners[0, 1]], 
                        [bev_corners[1, 0], bev_corners[1, 1]],
                        [bev_corners[2, 0], bev_corners[2, 1]],
                        [bev_corners[3, 0],bev_corners[3, 1]]])
    
    #[x1,y1,x2,y2]
    return points

# input : [x,y,w,l,yaw]
def calculate_iou(configs, b1, b2):
    background1 = np.zeros((configs.input_size[0], configs.input_size[1]))
    background2 = np.zeros((configs.input_size[0], configs.input_size[1]))

    # box : (x1, y1, x2, y2)
    box1_coord = get_points(b1)
    box2_coord = get_points(b2)

    box1 = cv2.fillConvexPoly(background1,box1_coord,1)
    box2 = cv2.fillConvexPoly(background2,box2_coord,1)

    inter = cv2.countNonZero(cv2.bitwise_and(box1,box2))
    union = cv2.countNonZero(cv2.bitwise_or(box1,box2))
    
    iou = inter/union
    
    return iou

def max_score(arrays, overlap_idx):
    max_score = 0
    max_score_idx = 0

    for idx in overlap_idx:
        score = arrays[idx][0]
        if score > max_score:
            max_score = score
            max_score_idx = idx

    remove_idx = list()
    for i in overlap_idx:
        if i != max_score_idx:
            remove_idx.append(i)

    return remove_idx

# detection box안에서 겹치는 것들을 제거해주는 함수(클래스가 다르면 제거하지 않음)
def remove_overlap_box(configs, arrays, threshold=0.5):
    check_dict = dict()

    for i, arr in enumerate(arrays):
        _score, _x, _y, _z, _h, _w, _l, _yaw = arr
        check_dict[i] = [_x, _y, _w, _l, _yaw]

    remove_idx_list = list()
    for a in range(len(check_dict) - 1):
        overlap_idx = list()

        for b in range(a+1, len(check_dict)):
            iou = calculate_iou(configs, check_dict[a], check_dict[b])
            if iou >= threshold:
                overlap_idx.append(a)
                overlap_idx.append(b)

        if len(overlap_idx) >= 2:
            overlap_idx = list(set(overlap_idx)) # 중복 제거
            remove_idx = max_score(arrays, overlap_idx)
            for i in remove_idx:
                remove_idx_list.append(i)

    remove_idx_list = list(set(remove_idx_list))

    boxes = np.array(())
    for i in range(len(arrays)):
        if i in remove_idx_list:
            continue
        else:
            if len(boxes) == 0:
                boxes = arrays[i].reshape(-1, 8)
            else:
                boxes = np.vstack((boxes, arrays[i])).reshape(-1, 8)

    # print('boxes : ', boxes)
    return boxes


# detection box안에서 겹치는 것들을 제거해주는 함수(클래스가 달라도 제거함)
def remove_overlap_box2(configs, detections, threshold=0.05):
    filtered_detections = dict()

    remove_idx_list = list()
    for i in range(configs.num_classes): # 0~4
        if len(detections[i]) != 0:
            for j in range(len(detections[i])): # len(detections[i])
                det1 = detections[i][j]

                for _i in range(configs.num_classes):
                    if len(detections[_i]) != 0:
                        for _j in range(len(detections[_i])):
                            if i ==_i:
                                if j < _j:
                                    det2 = detections[_i][_j]
                                else:
                                    continue
                            elif i < _i:
                                det2 = detections[_i][_j]
                            else:
                                continue

                            score1, x1, y1, z1, h1, w1, l1, yaw1 = det1
                            score2, x2, y2, z2, h2, w2, l2, yaw2 = det2

                            iou = calculate_iou(configs, [x1, y1, w1, l1, yaw1], [x2, y2, w2, l2, yaw2])

                            if iou >= threshold:
                                if score1 >= score2:
                                    remove_idx_list.append((_i, _j)) # class 번호, 클래스 내 index
                                else:
                                    remove_idx_list.append((i, j))

    remove_idx_list = list(set(remove_idx_list)) # 중복 제거

    # 클래스마다 제거해야 할 요소가 2개 이상인 경우에는 (class_num, [제거해야 할 인덱스들]) 형태로 뒤 인자를 list로 묶어준다.
    cnt_dict = dict()
    for idx, c_num in enumerate(zip(*remove_idx_list)):
        c_num = list(c_num)

        for i in range(configs.num_classes):
            cnt = c_num.count(i)
            if cnt == 0:
                continue
            else:
                cnt_dict[i] = cnt
        break

    for k in cnt_dict:
        if cnt_dict[k] >= 2:
            d_list = list()
            for idx, i in enumerate(remove_idx_list):
                if k == i[0]:
                    d_list.append(i[1])
                    

            del_list = (k, d_list)
            remove_idx_list.append(del_list)
            
            for j in d_list:
                remove_idx_list.remove((k, j))
    
    _detections = detections.copy()
    for k in remove_idx_list:
        c, idx = k[0], k[1]
        _detections[c] = np.delete(_detections[c], idx, axis=0)

    return _detections
