import os
import json

json_file =r"C:\Users\haidu\Desktop\YOLOP\YoloP_data\all_data_filter\dec_label\0000f77c-62c2a288.json"

dir_input = r"C:\Users\haidu\Desktop\YOLOP\YoloP_data\all_data_filter\image\train"
dir_output = r"C:\Users\haidu\Desktop\YOLOP\YoloP_data\all_data_filter\dec_label\train"

dict_file = {}
files = os.listdir(dir_input)
for fname in files:
    out_file = os.path.join(os.path.abspath(dir_output), fname.replace('.png', '.json'))
    with open(json_file, "r+", encoding="utf-8") as f:
        js = json.load(f)
        dict_file =js
    with open(out_file, 'w') as r:
        json.dump(dict_file, r)


