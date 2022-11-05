import os
from PIL import Image

road_dir_resize = r"C:\\Users\\haidu\\Desktop\\temp\\road_label_resize"
dest_dir = r"C:\\Users\\haidu\\Desktop\\temp\\road_label"

wide_road = os.listdir(r"C:\Users\haidu\Desktop\temp\wide_road")
wide_road = set(wide_road)

road_dir_files = os.listdir(road_dir_resize)
for file in road_dir_files:
    if file  in wide_road :
        child = os.path.join(road_dir_resize, file)       
        img = Image.open(child)
        dest = os.path.join(dest_dir,file)
        img.save(dest)