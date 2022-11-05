# -*- coding:utf-8 -*-
import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import argparse
import torch
import torch.nn.modules
import numpy as np
import onnx
import onnxruntime

from lib.config import cfg
from lib.models import get_net
from lib.utils.utils import create_logger, select_device, time_synchronized


def pth2onnx(cfg,opt):
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'pth2onnx')
    model = get_net(cfg)
    device = select_device(logger, opt.device) # ONNX models must be exported on CPU device for now. 
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    batch_size =opt.batchsize
    PRECISION = np.float32
    cfg.MODEL.NOT_INIT_MODEL[0] = True  #执行inference
    #伪造一个输入
    dummy_input = torch.randn(batch_size, 3, 384, 640, requires_grad=True).to(device)
    # torch_out = model(dummy_input)
    # print(torch_out)
    
    # Now let's try to export it an onnx format
    model_path = os.path.join(opt.save_dir,"YoloP_cpu.onnx")
    torch.onnx.export(model, dummy_input, model_path,
                        export_params=True, opset_version=11,
                        do_constant_folding= True, # 是否执行常量折叠优化
                        input_names = ['input'],
                        output_names = ['output'],
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}}
                        )

def to_numpy(tensor):
    if tensor.requires_grad :
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()

def verifyModel():
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'pth2onnx')
    model = get_net(cfg)
    device = select_device(logger, opt.device) # ONNX models must be exported on CPU device for now. 
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    batch_size =opt.batchsize
    # cfg.MODEL.NOT_INIT_MODEL[0] = True  #执行inference
    dummy_input = torch.randn(batch_size, 3, 384, 640, requires_grad=True).to(device)
    detect_out_tr, road_seg_out_tr, lane_seg_out_tr = model(dummy_input)
    dect_inf_out, dect_train_out = detect_out_tr
    torch_out = dect_inf_out, dect_train_out[0],dect_train_out[1],dect_train_out[2], road_seg_out_tr, lane_seg_out_tr
    # print (torch_out)

    onnx_model = onnx.load(r"C:\Users\haidu\Desktop\YOLOP\weights\YoloP_cpu.onnx") #加载保存的模型并输出一个onnx.ModelProto结构
    onnx.checker.check_model(onnx_model) #验证模型的结构
    #ONNX运行时运行模型
    ort_session = onnxruntime.InferenceSession(r"C:\Users\haidu\Desktop\YOLOP\weights\YoloP_cpu.onnx", \
                                                providers=["CUDAExecutionProvider"])

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # compare ONNX Runtime and PyTorch results
    for i in range(len(ort_outs)):
        np.testing.assert_allclose(to_numpy(torch_out[i]), ort_outs[i], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

def runOnnxModel():
    from PIL import Image
    import torchvision.transforms as transforms
    
    img = Image.open(r"C:\Users\haidu\Desktop\YOLOP\inference\images\det_val\001016.jpg")
    resize = transforms.Resize([384, 640])
    img = resize(img)
    
    # img_ycbcr = img.convert('YCbCr')
    # #将图像分割成Y、Cb和Cr三个分量,灰度图像(Y)，色度分量蓝差(Cb)和红差(Cr)。对于人眼来说，Y分量更敏感
    # img_y, img_cb, img_cr = img_ycbcr.split()

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img)
    img_y.unsqueeze_(0) #灰度调整后的图像的张量
    ort_session = onnxruntime.InferenceSession(r"C:\Users\haidu\Desktop\YOLOP\weights\YoloP_cpu.onnx", \
                                                providers=["CUDAExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[5]
    #输出图像
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
    img_out_y.show()
    # final_img = Image.merge(
    #     "YCbCr", [
    #         img_out_y,
    #         img_cb.resize(img_out_y.size, Image.BICUBIC),
    #         img_cr.resize(img_out_y.size, Image.BICUBIC),
    #     ]).convert("RGB")
    # final_img.save(r"C:\Users\haidu\Desktop\YOLOP\output\images\onnx_001016.jpg")


def genEngine():
    '''convert the ONNX model to a TRT engine using trtexec'''
    onnx_path = os.path.join(opt.save_dir,"YoloP_cpu.onnx")
    engin_path = os.path.join(opt.save_dir,"YoloP_cpu_int8.trt")
    io_precision = 'int8'   #输入或输出张量的精度 "fp32"|"fp16"|"int32"|"int8"
    mod_precision = 'int8'  #模型的量化精度  "fp32"|"fp16"|"int8"|"best"
    if io_precision and mod_precision:
        os.system('trtexec --onnx='+onnx_path+' --saveEngine='+engin_path+'  --workspace=20096 --explicitBatch \
        --inputIOFormats='+io_precision+':chw --outputIOFormats='+io_precision+':chw --'+ mod_precision+'\
        --useCudaGraph=True')
    else:
        os.system('trtexec --onnx='+ onnx_path +' --saveEngine='+engin_path +'  --workspace=20096 --explicitBatch \
        --useCudaGraph=True')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'C:\Users\haidu\Desktop\YOLOP\weights\model_best.pth', help='model.pth path(s)')
    parser.add_argument('--batchsize', type=int, default=1, help='set the batch size during the original export process to ONNX')
    parser.add_argument('--device', default='cpu', help=' cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_dir', type=str, default=r'C:\Users\haidu\Desktop\YOLOP\weights', help='directory to save results')
    parser.add_argument('--precision', default='np.float32', help='TensorRT supports TF32, FP32, FP16, and INT8 precisions.')
    opt = parser.parse_args()
    # pth2onnx(cfg, opt)
    # print("Finish Export ONNX")
    # verifyModel()
    # print("Finish verify model")
    # runOnnxModel()
    # print("Finish run Onnx mnodel")
    genEngine()
    print("Finish Export Engin")
    
    
