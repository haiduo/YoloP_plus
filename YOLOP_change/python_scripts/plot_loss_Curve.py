from itertools import count
import numpy as np
import matplotlib.pyplot as plt

TrainLossPath = r'C:\Users\haidu\Desktop\YOLOP\python_scripts\files\cha_train_seg_log.txt'

#2021-11-15 10:50:17,692 Epoch:[110][0/73] Len_Time:4.57509s 
#l_box:0.14469 l_obj:8.47914 l_cls:1.52969 l_road:0.00000 l_lane:0.00000 l_lane_iou:0.00000 Loss:10.15352
count_line = 0
loss_value = [[],[],[],[]]
with open(TrainLossPath, encoding="utf-8") as f:
    for line in f:
        count_line = count_line+1
        line = line.split('s ')[1].split()
        loss_value[0].append(float(line[3].split(':')[1]))
        loss_value[1].append(float(line[4].split(':')[1]))
        loss_value[2].append(float(line[5].split(':')[1]))
        loss_value[3].append(float(line[6].split(':')[1]))

epoch = np.arange(1, count_line+1)/5

loss_valus = np.array(loss_value)*100

plt.figure(num = 'Train', figsize = (8,6), dpi = 300)
ax = plt.subplot(1,1,1)
plt.title('Segmentation_Train')

plt.plot(epoch, loss_valus[0], 'r', label='L_road', linewidth=1)
plt.plot(epoch, loss_valus[1], 'b', label='L_lane', linewidth=1)
plt.plot(epoch, loss_valus[2], 'g', label='L_iou', linewidth=1)
plt.plot(epoch, loss_valus[3], 'm', label='L_seg', linewidth=1)

plt.xlim(epoch.min()*0.01, epoch.max()*1.01)
plt.ylim(np.array([loss_valus.min()]).min()*0.9, np.array([loss_valus.max()]).max()*1.1)

plt.xlabel(u'epoch')
plt.ylabel(u'Loss(%)')
plt.legend()
figPath = r'C:\Users\haidu\Desktop\YOLOP\python_scripts\files\cha_train_seg_log.png'
plt.savefig(figPath, dpi=300)
plt.show()

















