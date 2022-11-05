import os
import json

dict ={}
filedir = './input'
outdir = './input_1'
fileNames = os.listdir(filedir)  #获取当前目录下的所有文件
for f in fileNames:
    check = os.path.join(os.path.abspath(filedir),f)
    if os.path.isfile(check):
        with open(check,'r+',encoding="utf-8") as file:
            js = json.load(file)
            for key in js:
                if key == "imagePath":        
                    js[key] = js[key].replace("\\冯志远标注\\","\\")
            dict = js
    newfile = os.path.join(os.path.abspath(outdir),f)
    with open(newfile,'w') as r:  
        json.dump(dict ,r)       
 