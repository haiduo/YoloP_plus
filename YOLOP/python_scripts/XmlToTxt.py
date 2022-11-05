"""Convert Pascal VOC to YOLOv5 darknet format
"""
import xml.etree.ElementTree as ET
import os
from os import getcwd
from os.path import join

sets = ['train', 'valid', 'test']
classes = ['Crack', 'Net', 'AbnormalManhole', 'Pothole', 'Marking']  # 自己训练的类别
abs_path = getcwd()
print(abs_path)

def convert(size, box):
    dw = 1. / (size[0] - 1)     # 这块减1操作 也可以不减，根据比赛要求来变化
    dh = 1. / (size[1] - 1)
    x = (box[0] + box[1]) / 2.0   # 这块减1操作 也可以不减，试验的，为了更好的回归框
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(image_id):
    in_file = open('./seed/annotations/%s.xml' % (image_id), encoding="utf-8")
    out_file = open('./seed/labels/%s.txt' % (image_id), 'w', encoding="utf-8")
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult) == 1:
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b1 < 0:
            b1 = 0
        if b3 < 0:
            b3 = 0
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)

        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# def getFiles(base):
#     filenames = os.listdir(base)
#     with open('train.txt', 'w', encoding="utf-8") as f:
#         for row in filenames:
#             f.write(row + "\n")

for image_set in sets:
    if not os.path.exists('seed/test/labels/'):
        os.makedirs('seed/test/labels/')

    image_ids = open('./seed/ImageSets/Main/%s.txt' % (image_set), encoding="utf-8").read().strip().split()
    list_file = open('./%s.txt' % (image_set), 'w', encoding="utf-8")
    for image_id in image_ids:
        # list_file.write(join(abs_path + '\seed\images\%s.jpg\n' % (image_id)))
        list_file.write(join(abs_path + '\seed\images\%s.jpg' % (image_id)+' ' + abs_path + '\seed\\annotations\%s.xml\n' % (image_id)))
        convert_annotation(image_id)
    list_file.close()

