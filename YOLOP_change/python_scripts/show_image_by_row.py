import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片 https://blog.csdn.net/Strive_For_Future/article/details/81264096
#from PIL import Image #https://blog.csdn.net/Tang_Klay/article/details/110384726
import numpy as np
 
plt.figure(num = 'cluster', figsize = (5,8), dpi = 100)
k=1
for i in range(1,6):
    for j in range(1,9):
        path = '/home/haiduo/code/animal_human_kp-master/checkdata/'+str(i)+'0'+str(j)+'.jpg'
        #pil_im = Image.open(path)
        pil_im = mpimg.imread(path)
        plt.subplot(5,8,k)
        k=k+1
        plt.imshow(pil_im)
        plt.axis('off') # 不显示坐标轴

# for i in range(16):
#     path = '/home/haiduo/code/data/checkImages/TCNN/0'+str(i)+'.jpg'
#     #pil_im = Image.open(path)
#     pil_im = mpimg.imread(path)
#     plt.subplot(6,8,i+1+16)
#     plt.imshow(pil_im)
#     plt.axis('off') # 不显示坐标轴

# for i in range(16):
#     path = '/home/haiduo/code/data/checkImages/STCNN/0'+str(i)+'.jpg'
#     #pil_im = Image.open(path)
#     pil_im = mpimg.imread(path)
#     plt.subplot(6,8,i+1+32)
#     plt.imshow(pil_im)
#     plt.axis('off') # 不显示坐标轴

figPath = '/home/haiduo/code/allanimaltestResults.png'
plt.savefig(figPath, dpi=150)
plt.show()