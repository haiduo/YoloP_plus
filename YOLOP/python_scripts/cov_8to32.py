import os
import numpy as np
from PIL import Image

'''
参考：https://blog.csdn.net/weixin_44617502/article/details/113567148
另一种方法：https://www.coder.work/article/1247090
'''

path = r'C:\Users\haidu\Desktop\all_data_sorc\label'
lane_path = r'C:\Users\haidu\Desktop\all_data_sorc\lane_label'
road_path = r'C:\Users\haidu\Desktop\all_data_sorc\road_label'

files = os.listdir(path)
for fname in files:
    files = os.path.join(path, fname)
    img = Image.open(files)
    img_array = np.array(img)   # 把图像转成数组格式
    shape = img_array.shape
    # print(img_array.shape)
    dst_road = np.zeros((shape[0], shape[1]))
    dst_lane = np.zeros((shape[0], shape[1]))
    # color =set()
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            value = img_array[i, j]
            # if value not in color:
            #    color.add(value)
            if img_array[i, j] == 1:
                dst_lane[i, j] = 255
                dst_road[i, j] = 255
            if img_array[i, j] == 2:
                dst_road[i, j] = 255
    img_road = Image.fromarray(np.uint8(dst_road))
    img_lane = Image.fromarray(np.uint8(dst_lane))
    # img_road = Image.fromarray(dst_road, mode='RGB')
    # img_lane = Image.fromarray(dst_lane, mode='RGB')
    # img_road = Image.open(img_road).convert('RGBA')
    # img_lane = Image.open(img_lane).convert('RGBA')
    file_name, file_extend = os.path.splitext(fname)
    dst_road = os.path.join(os.path.abspath(road_path), file_name + '.png')
    dst_lane = os.path.join(os.path.abspath(lane_path), file_name + '.png')
    img_road.save(dst_road)
    img_lane.save(dst_lane)
