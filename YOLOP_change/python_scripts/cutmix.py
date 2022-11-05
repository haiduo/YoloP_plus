
import matplotlib.pyplot as plt
import numpy as np
import cv2

'''参考：https://www.cnblogs.com/hikari-1994/p/15142301.html'''
plt.rcParams['figure.figsize'] = [10, 10]

def rand_bbox(size, lamb):
    """
    生成随机的bounding box
    :param size:
    :param lamb:
    :return:
    """
    W = size[0]
    H = size[1]

    # 得到一个bbox和原图的比例
    cut_ratio = np.sqrt(1.0 - lamb)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    # 得到bbox的中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(image_batch, image_batch_labels, alpha=1.0):
    # 决定bbox的大小，服从beta分布
    # 参考https://zhuanlan.zhihu.com/p/149964631
    lam = np.random.beta(alpha, alpha)
    #  permutation: 如果输入x是一个整数，那么输出相当于打乱的range(x)
    rand_index = np.random.permutation(len(image_batch))

    # 对应公式中的y_a,y_b
    target_a = image_batch_labels
    target_b = image_batch_labels[rand_index]

    # 根据图像大小随机生成bbox
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_batch[0].shape, lam)

    image_batch_updated = image_batch.copy()

    # image_batch的维度分别是 batch x 宽 x 高 x 通道
    # 将所有图的bbox对应位置， 替换为其他任意一张图像
    # 第一个参数rand_index是一个list，可以根据这个list里索引去获得image_batch的图像，也就是将图片乱序的对应起来
    image_batch_updated[:, bbx1: bbx2, bby1:bby2,
                        :] = image_batch[rand_index, bbx1:bbx2, bby1:bby2, :]  # 目标和标签大小一样
    # 计算 1 - bbox占整张图像面积的比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / \
        (image_batch.shape[1] * image_batch.shape[2])
    # 根据公式计算label
    label = target_a * lam + target_b * (1. - lam)
    return image_batch_updated, label


if __name__ == '__main__':
    cat = cv2.cvtColor(cv2.imread(
        "background/hangzhou_resize_000605.png"), cv2.COLOR_BGR2RGB)
    dog = cv2.cvtColor(cv2.imread(
        "background/hangzhou_resize_000669.png"), cv2.COLOR_BGR2RGB)
    updated_img, label = cutmix(
        np.array([cat, dog]), np.array([[0, 1], [1, 0]]), 0.5)

    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False)
    #效果:(https://img2020.cnblogs.com/blog/1621431/202108/1621431-20210814233143289-396153337.png)
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax1.imshow(updated_img[0])
    ax2.imshow(updated_img[1])
    plt.show()
