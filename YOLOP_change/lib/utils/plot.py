## 处理pred结果的.json文件,画图
import matplotlib.pyplot as plt
import cv2, os
import numpy as np
import torch
from .visualize_boxes_on_img import visualize_boxes

def plot_img_and_mask(img, mask, index,epoch,save_dir):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(save_dir+"/batch_{}_{}_seg.png".format(epoch,index))

def show_seg_result(img_org, result, index, i, epoch, save_dir=None, is_ll=False,palette=None,is_demo=False,is_gt=False):
    if palette is None:
        palette = np.random.randint(0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    
    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)

        color_area[result[0] == 1] = [0, 255, 0]
        color_area[result[1] ==1] = [255, 0, 0]
        color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    h, w, _ = color_seg.shape
    h_org, w_org, _ = img_org.shape
    color_mask = np.mean(color_seg, 2)
    img_org = cv2.resize(img_org, (w, h), interpolation=cv2.INTER_LINEAR)
    img_org[color_mask != 0] = img_org[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img_org = img_org.astype(np.uint8)
    img_org = cv2.resize(img_org, (w_org, h_org), interpolation=cv2.INTER_LINEAR)
    
    if not is_demo:
        if is_gt:
            if is_ll:
                cv2.imwrite(save_dir+"/{}_{}_{}_lane_gt.png".format(epoch,index, i), img_org)
            else:
                cv2.imwrite(save_dir+"/{}_{}_{}_road_gt.png".format(epoch,index, i), img_org) 
        else:
            if is_ll:
                cv2.imwrite(save_dir+"/{}_{}_{}_lane_result.png".format(epoch,index, i), img_org)
            else:
                cv2.imwrite(save_dir+"/{}_{}_{}_road_result.png".format(epoch,index, i), img_org) 
    return img_org

def plot_one_box(box, img, cls, conf):
    # Plots one bounding box on image img
    # tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # color = color or [random.randint(0, 255) for _ in range(3)]
    # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # img = img.cpu().detach().numpy()
    box = box.cpu().detach().numpy()
    cls = cls.cpu().detach().numpy().astype(np.int16)
    if torch.is_tensor(conf):
        conf = conf.cpu().detach().numpy()
    cls = cls.flatten()
    conf = conf.flatten()
    # print("type(boxes):{}{}\ntype(scores):{}{}\ntype(classes):{}{}\ntype(image):{}{}".\
    # format(type(box), box.shape, type(img), img.shape, type(cls), cls.shape, type(conf), conf.shape))
    visualize_boxes(image=img, boxes=box, classes=cls, scores=conf)

if __name__ == "__main__":
    pass
