import sys
import cv2
import os
import xml.etree.ElementTree as ET

classes = ['Crack', 'Net', 'AbnormalManhole', 'Pothole', 'Marking']  # 自己训练的类别

# 遍历指定目录，显示目录下的所有文件名
def ResizeImage(imagedir, destimagedir):
    pathDir = os.listdir(imagedir)  # list all the path or file  in filepath
    for allDir in pathDir:
        child = os.path.join(imagedir, allDir)
        dest = os.path.join(destimagedir, allDir)
        img1 = cv2.imread(child)
        img2 = cv2.resize(img1, (640, 640))
        cv2.imwrite(dest, img2)

def ResizeLable(annodir, destannodir):
    pathDir = os.listdir(annodir)  # list all the path or file  in filepath
    # oldcls =''
    # newcls =''
    total = len(pathDir)
    for i, allDir in enumerate(pathDir):
        child = os.path.join(annodir, allDir)
        dest = os.path.join(destannodir, allDir)
        tree = ET.parse(child)
        root = tree.getroot()

        size = root.find('size')
        oldwidth = float(size.find('width').text)
        size.find('width').text = str(640)
        rate_x = 640 / oldwidth
        oldheight = float(size.find('height').text)
        size.find('height').text = str(640)
        rate_y = 640 / oldheight

        for obj in root.findall("object"):
            cls = obj.find('name').text
            xmlbox = obj.find("bndbox")
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            b1, b2, b3, b4 = b
            # 标注越界修正
            if b1 < 0:
                b1 = 0
            if b3 < 0:
                b3 = 0
            if b2 > oldwidth:
                b2 = oldwidth
            if b4 > oldheight:
                b4 = oldheight
            # 判断压缩后的标框是否合理
            w = b2 - b1
            h = b4 - b3
            if cls not in classes or w <=2 or h<=2:
                root.remove(obj)
                # print(child)
                continue
            xmlbox.find('xmin').text = str(int(b1 * rate_x))
            xmlbox.find('xmax').text = str(int(b2 * rate_x))
            xmlbox.find('ymin').text = str(int(b3 * rate_y))
            xmlbox.find('ymax').text = str(int(b4 * rate_y))
        # 写入修改信息，生成新的xml文件
        tree = ET.ElementTree(root)
        tree.write(dest, encoding="utf-8", xml_declaration=True)
        # 显示进度条
        process = int(i * 100 / total)
        s1 = "\r%d%%[%s%s]" % (process, "*" * process, " " * (100 - process))
        s2 = "\r%d%%[%s]" % (100, "*" * 100)
        sys.stdout.write(s1)
        sys.stdout.flush()
    sys.stdout.write(s2)
    sys.stdout.flush()
    print('')
    print('Resize is complete!')


if __name__ == '__main__':
    # imagedir = r'C:\Users\haidu\Desktop\CnnModel\datasets\seed\images_src'  # source images
    # destimagedir = r'C:\Users\haidu\Desktop\CnnModel\datasets\seed\images'  # resized images saved here
    annodir = r'C:\Users\haidu\Desktop\CnnModel\datasets\seed\annotations_src'  # source lables
    destannodir = r'C:\Users\haidu\Desktop\CnnModel\datasets\seed\annotations'  # resized lables saved here
    # ResizeImage(imagedir, destimagedir)
    ResizeLable(annodir, destannodir)

