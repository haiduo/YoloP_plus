# coding=utf-8
import json
import os
from PIL import Image

classes = ['Crack', 'Net', 'AbnormalManhole', 'Pothole', 'Marking']  # 自己训练的类别

global l_xmin, l_xmax, l_ymin, l_ymax
global l_xmin1, l_xmax1, l_ymin1, l_ymax1

l_xmin = []
l_xmax = []
l_ymin = []
l_ymax = []
l_xmin1 = []
l_xmax1 = []
l_ymin1 = []
l_ymax1 = []


def convert(size , box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = float(box[0]) / float(dw)
    y = float(box[1]) / float(dh)
    w = float(box[2]) / float(dw)
    h = float(box[3]) / float(dh)

    xmin = x - w / 2.0
    ymin = y - h / 2.0
    xmax = x + w / 2.0
    ymax = y + h / 2.0
    return [xmin, ymin, xmax, ymax]

def convert_annotation(image_id):
    if not os.path.exists('./seed/test/labels/%s.txt' % (image_id)):
        with open('./seed/test/result/%s.json' % (image_id), 'w') as dump_f:
            json.dump('', dump_f)
    else:
        in_file = open('./seed/test/labels/%s.txt' % (image_id), encoding="utf-8")
        img = Image.open('./seed/test/images/%s.jpg' % (image_id))

        w = img.width - 1  # 图片的宽
        h = img.height - 1  # 图片的高

        result = []
        # 逐行读取
        for line in in_file:
            lst = line.strip().split()
            box = convert((w, h), (lst[1], lst[2], lst[3], lst[4]))
            if (box[0] < 0):
                box[0] += 1
                if (image_id not in l_xmin):
                    l_xmin.append(image_id)
            if (box[1] < 0):
                box[1] += 1
                if (image_id not in l_ymin):
                    l_ymin.append(image_id)
            if (box[2] > w  ):
                box[2] -= 1
                if (image_id not in l_xmax):
                    l_xmax.append(image_id)
            if (box[3] > h  ):
                box[3] -= 1
                if (image_id not in l_ymax):
                    l_ymax.append(image_id)
            item = {
                "category": classes[int(lst[0])],
                "xmin": box[0],
                "ymin": box[1],
                "xmax": box[2],
                "ymax": box[3],
                "score": float(lst[5])
            }
            if (box[0] < 0):
                if (image_id not in l_xmin1):
                    l_xmin1.append(image_id)
            if (box[1] < 0):
                if (image_id not in l_ymin1):
                    l_ymin1.append(image_id)
            if (box[2] > w  ):
                if (image_id not in l_xmax1):
                    l_xmax1.append(image_id)
            if (box[3] > h  ):
                if (image_id not in l_ymax1):
                    l_ymax1.append(image_id)
            result.append(item)
        # 关闭文件
        in_file.close()
        # 写入数据
        with open('./seed/test/result/%s.json' % (image_id), 'w', encoding="utf-8") as dump_f:
            json.dump(result, dump_f)

def getFiles(base):
    filenames = os.listdir(base)
    with open('test.txt', 'w', encoding="utf-8") as f:
        for row in filenames:
            f.write(row + "\n")



def main():
    base = './seed/test/images/'
    getFiles(base)  # get all files name

    if not os.path.exists('seed/test/result'):
        os.makedirs('seed/test/result')

    image_ids = open('test.txt')
    for image_id in image_ids:
        image_id = image_id.split('.')[0]
        convert_annotation(image_id)
    image_ids.close()


main()
print(len(l_xmin), l_xmin)
print(len(l_ymin), l_ymin)
print(len(l_xmax), l_xmax)
print(len(l_ymax), l_ymax)
print("")
print(len(l_xmin1), l_xmin1)
print(len(l_ymin1), l_ymin1)
print(len(l_xmax1), l_xmax1)
print(len(l_ymax1), l_ymax1)
