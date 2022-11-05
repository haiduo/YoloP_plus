import os
import cv2
import numpy as np
import json
from tqdm import tqdm
"""
生成裁剪区域
输入为：样本的size和生成的随机lamda值
"""
#json_label: 10:accident, 11:fire, 12:dense_traffic, 13:chemicals_vehicle, 14:bus, 15:road_toss 

label = {"accident":7, "fire":8, "dense_traffic":9, "chemicals_vehicle":10, "bus":5, "road_toss":11}

base_dir = r"/root/traffic/traffic_events_dangerous"

def gen_json(result_boxs, name_background, dest_annotation_dir, batch):

    (_, name) = os.path.split(name_background)  # 分割文件目录/文件名和后缀
    #base_dir = os.path.abspath(os.path.dirname(base_dir)) #获取上级目录
    (_, name_extension) = os.path.splitext(name_background)
    name = name.split(name_extension)[0]+'_'+str(batch)
    file_name = os.path.join(dest_annotation_dir, name+'.json')
    
    items = {
        "name": name,
        "frames": [
            {
                "objects": result_boxs
                
            }
        ]
    }
    with open(file_name, 'w') as dump_f:
        json.dump(items, dump_f)

def mixup(name_background, name_background_mask, target_batch, n_holes, batch, dest_annotation_dir, alpha=1.0):
    
    image_background = cv2.imread(name_background)
    image_background = cv2.resize(image_background, (1280, 720))
    h_background, w_background, _ = image_background.shape

    child_background_mask = cv2.imread(name_background_mask)
    img_mask = np.array(child_background_mask)

    func_mask = lambda x,y:True if img_mask[x][y][0]==255 else False #判断x,y是否落在道路区域

    result_boxs = []
    update_background = image_background.copy()
    for _ in range(n_holes):
        
        rand_index = np.random.randint(len(target_batch))
        target = target_batch[rand_index]
        # (_, name_extension) = os.path.splitext(name_background)
        target_name = target.split(' ')[0]
        cls = label[target_name]
        image_target = cv2.imread(os.path.join(base_dir,target))
        image_target = cv2.resize(image_target, (np.random.randint(120,150), np.random.randint(50,80)))

        h_target, w_target, _ = image_target.shape
        #裁剪目标图像区域的中心点
        while(1):
            x, y = np.random.randint(w_target, w_background-w_target), np.random.randint(h_target, h_background-h_target)
            if func_mask(y, x) == True:
                break
        
        #目标区域转化为x1,y1,x2,y2形式
        x1 = np.clip(x - w_target // 2, 0, w_background)
        x2 = np.clip(x + w_target // 2, 0, w_background)
        y1 = np.clip(y - h_target // 2, 0, h_background)
        y2 = np.clip(y + h_target // 2, 0, h_background)

        box = (x1, y1, x2, y2)
        img_org = update_background[y1:y2, x1:x2, :]
        img_mix = image_target[:y2-y1, :x2-x1, :]

        lam = np.random.beta(alpha, alpha)
        scale_pixel = np.random.randint(np.floor(0.1 * w_target))
        h_mask, w_mask, _ = img_mix.shape
        img = lam * img_org + (1 - lam) * img_mix
        img[scale_pixel:h_mask-scale_pixel, scale_pixel:w_mask-scale_pixel, :] = img_mix[scale_pixel:h_mask-scale_pixel, scale_pixel:w_mask-scale_pixel, :] 
        update_background[y1:y2, x1:x2, :] = img
        #将标定框写进字典，方便后面写入json文件  
        item = {
            "category": target_name,
            "id": cls,
            "box2d": {
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3])
            }
        }
        result_boxs.append(item)
    gen_json(result_boxs, name_background, dest_annotation_dir, batch)

    return update_background


def cutout(name_background, name_background_mask, target_batch, n_holes, dest_annotation_dir):
    
    image_background = cv2.imread(name_background)
    image_background = cv2.resize(image_background, (1280, 720))
    h_background, w_background, _ = image_background.shape

    child_background_mask = cv2.imread(name_background_mask)
    img_mask = np.array(child_background_mask)

    func_mask = lambda x,y:True if img_mask[x][y][0]==255 else False #判断x,y是否落在道路区域

    result_boxs = []
    update_background = image_background.copy()
    for _ in range(n_holes):
        
        rand_index = np.random.randint(len(target_batch))
        target = target_batch[rand_index]
        cls = label[target]
        image_target = cv2.imread(os.path.join(base_dir,target))
        image_target = cv2.resize(image_target, (150, 100))

        h_target, w_target, _ = image_target.shape
        #裁剪目标图像区域的中心点
        while(1):
            x, y = np.random.randint(w_target, w_background-w_target), np.random.randint(h_target, h_background-h_target)
            if func_mask(y, x) == True:
                break

        #目标区域转化为x1,y1,x2,y2形式
        x1 = np.clip(x - w_target // 2, 0, w_background)
        x2 = np.clip(x + w_target // 2, 0, w_background)
        y1 = np.clip(y - h_target // 2, 0, h_background)
        y2 = np.clip(y + h_target // 2, 0, h_background)

        box = (x1, y1, x2, y2)
        update_background[y1:y2, x1:x2, :] = image_target[:y2-y1, :x2-x1, :]
        #将标定框写进字典，方便后面写入json文件  
        item = {
            "category": label[cls],
            "id": cls,
            "box2d": {
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3])
            }
        }
        result_boxs.append(item)
    gen_json(result_boxs, name_background, dest_annotation_dir)

    return update_background


if __name__ == '__main__':
    imagedir_target = r'/root/traffic/traffic_events_dangerous'  # source images
    imagedir_background = r'/root/traffic/wide_road'
    imagedir_background_mask = r"/root/traffic/road_label"

    dest_image_dir = r'/root/traffic/dataset_traffic/images'  # images saved here
    dest_annotation_dir = r"/root/traffic/dataset_traffic/annotations"
    # 遍历指定目录，显示目录下的所有文件名
    pathDir_target = os.listdir(imagedir_target)  # list all the path or file  in filepath
    pathDir_background = os.listdir(imagedir_background) 
    for i in range(3):
        for background in tqdm(pathDir_background):

            child_background = os.path.join(imagedir_background, background)
            child_background_mask = os.path.join(imagedir_background_mask, background)
            
            (_, name_extension) = os.path.splitext(background)
            background_name = background.split(name_extension)[0]+'_'+str(i)+name_extension
            # updated_img = cutout(child_background, child_background_mask, pathDir_target, 1, dest_annotation_dir)
            updated_img = mixup(child_background, child_background_mask, pathDir_target, 2, i, dest_annotation_dir, alpha=1.0)
            cv2.imwrite(os.path.join(dest_image_dir, background_name), updated_img)