import os
import shutil

label_file =r"/root/YOLOP/data/segementation/lane_label/train/guangzhou_resize_000010.png"

dir_input = r"/root/YOLOP/data/detection/images/train"
dir_output = r"/root/YOLOP/data/detection/lane_label/train"

dict_file = {}
files = os.listdir(dir_input)
for fname in files:
    (_, name) = os.path.split(fname)  # 分割文件目录/文件名和后缀
    (_, name_extension) = os.path.splitext(fname)
    name = name.split(name_extension)[0]
    file_path = os.path.join(dir_output, name+'.png')

    shutil.copyfile(label_file, file_path)
    
