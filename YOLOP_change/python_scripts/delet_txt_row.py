import os
del_dir=r"C:\Users\haidu\Desktop\YOLOP\python_scripts\files"           #要处理文件的目录
filelist=os.listdir(del_dir)       #提取文件名存放在filelist中
for file in filelist:               #遍历文件名
    del_file = del_dir+'\\'+file       #程序和文件不在同一目录下要用绝对路径
    save_file = del_dir+'\\cha_'+file
    lines=[a for a in open(del_file,"r") if((a.find("Driving area Segment")==-1) and (a.find("Lane line Segment")==-1)  
    and  (a.find("Detect")==-1) and (a.find("inference")==-1) and (a.find("saving")==-1) )]       #注意这里要用and
#把文件中含有1234、5678和abcd的行删掉
    fd=open(save_file,"w")
    fd.writelines(lines)
    fd.close