# 최악의 시간 복잡도..임의로 빨리 만든 파일(5시간 넘게 걸림)
import os
import shutil
import numpy as np
import re
import sys
import time

start_time = time.time()

# 시작 파일 번호 정하기
criterion_file_list = os.listdir('./c1')
criterion_file_list.sort()

criterion_num = int(criterion_file_list[0].split('.')[0])

for i in range(2, 9):
    if i <= 4:
        compare_file_list = os.listdir(f'./c{str(i)}')
        compare_file_list.sort()

        compare_num = int(compare_file_list[0].split('.')[0])
        if criterion_num > compare_num:
            criterion_num = compare_num

    else:
        compare_file_list = os.listdir(f'./l{str(i-4)}')
        compare_file_list.sort()

        compare_num = int(compare_file_list[0].split('.')[0])
        if criterion_num > compare_num:
            criterion_num = compare_num

# 파일 나누기
for criterion_file in criterion_file_list: # c1폴더의 file list를 순회
    criterion_file_name = criterion_file.split('.')[0][:-2]
    p = re.compile(criterion_file_name + '\d\d' + '[.][a-zA-Z][a-zA-Z]')

    # c2~c4, l1~l4 폴더에 c1에 있는 파일과 같은 timestmap의 파일이 전부 있는지 확인
    cam_n, lid_n = 0, 0
    for i in range(2, 9):
        if i <= 4:
            c_file_path = f'./c{str(i)}'
            c_file_list = os.listdir(c_file_path)
            
            for f in c_file_list:        
                if p.match(f):
                    cam_n += 1    
                    break
            
        else:
            l_file_path = f'./l{str(i-4)}'
            l_file_list = os.listdir(l_file_path)
            
            for f in l_file_list:        
                if p.match(f):
                    lid_n += 1    
                    break
        
    # c2~c4, l1~l4 폴더에 c1에 있는 파일과 같은 timestmap의 파일이 전부 있다면, 각 파일을 sync 폴더에 저장
    if cam_n == 3 and lid_n == 4:
        for i in range(1, 9):
            if i <= 4:
                c_file_path = f'./c{str(i)}'
                c_dst_path = os.path.join('./sync', f'c{str(i)}')
                if not os.path.exists(c_dst_path):
                    os.makedirs(c_dst_path)

                c_file_list = os.listdir(c_file_path)
                for f in c_file_list:        
                    if p.match(f):
                        shutil.copyfile(os.path.join(c_file_path, f), os.path.join(c_dst_path, f))
                        break       

            else:
                l_file_path = f'./l{str(i-4)}'
                l_dst_path = os.path.join('./sync', f'l{str(i-4)}')
                if not os.path.exists(l_dst_path):
                    os.makedirs(l_dst_path)

                l_file_list = os.listdir(l_file_path)
                for f in l_file_list:        
                    if p.match(f):
                        shutil.copyfile(os.path.join(l_file_path, f), os.path.join(l_dst_path, f))
                        break

end_time = time.time()
print('exec time : ' + int(end_time - start_time) // 60 + 'min', int(end_time - start_time) % 60 + "sec")