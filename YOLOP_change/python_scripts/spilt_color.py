from PIL import Image
import numpy as np

img = Image.open(r"C:\Users\haidu\Desktop\YOLOP\YoloP_data\guangzhou_resize_000010.png")
# img.show()
img_array = np.array(img)  # 把图像转成数组格式img = np.asarray(image)
shape = img_array.shape
# print(img_array.shape)
dst_road = np.zeros((shape[0], shape[1]))
dst_lane = np.zeros((shape[0], shape[1]))
# color =set()
for i in range(0, shape[0]):
    for j in range(0, shape[1]):
        value = img_array[i, j]
        # if value not in color:
        #    color.add(value)
        if img_array[i, j] == 1:
            dst_lane[i, j] = 255
            dst_road[i, j] = 255
        if img_array[i, j] == 2:
            dst_road[i, j] = 255

img_road = Image.fromarray(np.uint8(dst_road))
img_lane = Image.fromarray(np.uint8(dst_lane))

# img_load.show()
img_road.save("road.png", "png")
img_lane.save("lane.png", "png")
