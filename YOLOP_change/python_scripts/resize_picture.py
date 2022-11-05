import cv2
import os


# 遍历指定目录，显示目录下的所有文件名
def ResizeImage(imagedir, destimagedir):
    pathDir = os.listdir(imagedir)  # list all the path or file  in filepath
    for allDir in pathDir:
        child = os.path.join(imagedir, allDir)
        dest = os.path.join(destimagedir, allDir)
        img1 = cv2.imread(child)
        img2 = cv2.resize(img1, (1280, 720))
        cv2.imwrite(dest, img2)

if __name__ == '__main__':
    imagedir = r'C:\Users\haidu\Desktop\all_data_sorc\road_label'  # source images
    destimagedir = r'C:\Users\haidu\Desktop\all_data_sorc\road_label_resize'  # resized images saved here
    ResizeImage(imagedir, destimagedir)