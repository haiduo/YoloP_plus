import os
import json

json_file =r"/root/YOLOP/data/detection/annotations/train/guangzhou_resize_000010_0.json"

dir_input = r"/root/YOLOP/data/segementation/images/val"
dir_output = r"/root/YOLOP/data/segementation/annotations/val"

dict_file = {}
files = os.listdir(dir_input)
for fname in files:
    out_file = os.path.join(os.path.abspath(dir_output), fname.replace('.png', '.json'))
    with open(json_file, "r+", encoding="utf-8") as f:
        js = json.load(f)
        dict_file =js
    with open(out_file, 'w') as r:
        json.dump(dict_file, r)


