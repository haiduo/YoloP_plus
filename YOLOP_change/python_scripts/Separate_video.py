'''
批量提取视频的所有帧
'''
import os
import cv2

# 存放视频的地址
videos_src_path = r'C:\Users\haidu\Desktop\YOLOP\inference/videos\hangzhou'
# 存放图片的地址
videos_save_path = r'C:\Users\haidu\Desktop\YOLOP\inference/videos\images'
# 返回videos_src_path路径下包含的文件或文件夹名字的列表（所有视频的文件名），按字母顺序排序
videos = os.listdir(videos_src_path)

for each_video in videos:
    # 获取每个视频的名称
    each_video_name, _ = each_video.split('.')
    # 创建目录，来保存图片帧
    os.mkdir(videos_save_path + '/' + each_video_name)
    # 获取保存图片的完整路径，每个视频的图片帧存在以视频名为文件名的文件夹中
    each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'
    # 获取每个视频的完整路径
    each_video_full_path = os.path.join(videos_src_path, each_video)
    # 读入视频
    cap = cv2.VideoCapture(each_video_full_path)
    # 设置提取视频的第几帧(n=几，就提取第几帧)
    n = 10
    i = 0
    # 统计提取视频帧的数量
    frame_count = 1
    success = True
    while (success):
        # 提取视频帧，success为是否成功获取视频帧（true/false），第二个返回值为返回的视频帧
        success, frame = cap.read()
        i += 1
        if success == True and i % n == 0:
            # 存储视频帧
            cv2.imwrite(each_video_save_full_path + "%06d.jpg" % frame_count, frame)
            frame_count = frame_count + 1

    print("The number of video frames:", frame_count)
