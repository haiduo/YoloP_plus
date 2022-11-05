import os
import numpy as np
import shutil


img_dir = r"/root/YOLOP/data/detection/images/train"
annotaions_dir = r"/root/YOLOP/data/detection/annotations/train"
lane_dir = r"/root/YOLOP/data/detection/lane_label/train"
road_dir = r"/root/YOLOP/data/detection/road_label/train"

dest_img = r"/root/YOLOP/data/detection_s/images/train"
dest_annotations = r"/root/YOLOP/data/detection_s/annotations/train"
dest_lane = r"/root/YOLOP/data/detection_s/lane_label/train"
dest_road = r"/root/YOLOP/data/detection_s/road_label/train"

dir_files = os.listdir(img_dir)
result = set()
while len(result) < 500:
    rand_index = np.random.randint(len(dir_files))
    (_, name) = os.path.split(dir_files[rand_index])  # 分割文件目录/文件名和后缀
    (_, name_extension) = os.path.splitext(dir_files[rand_index])
    name = name.split(name_extension)[0]

    if rand_index not in result:
        shutil.copy(os.path.join(img_dir, dir_files[rand_index]),dest_img) #将文件f1 移动到 f2目录下
        shutil.copy(os.path.join(annotaions_dir, name+'.json'),dest_annotations) #将文件f1 移动到 f2目录下
        shutil.copy(os.path.join(lane_dir, name+'.png'),dest_lane) #将文件f1 移动到 f2目录下
        shutil.copy(os.path.join(road_dir, name+'.png'),dest_road) #将文件f1 移动到 f2目录下
        result.add(rand_index)      

    