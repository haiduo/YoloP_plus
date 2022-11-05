import argparse
import os, sys
import time
from pathlib import Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import torchvision.transforms as transforms

from lib.config import cfg
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
#############
from collections import OrderedDict
from torch.autograd import Variable
import json
import torch.nn as nn
import numpy as np
#############

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform=transforms.Compose([transforms.ToTensor(),normalize,])

#######################################
def get_output_size(summary_dict, output):
    if isinstance(output, tuple):
        for i in range(len(output)):
            summary_dict[i] = OrderedDict()
            summary_dict[i] = get_output_size(summary_dict[i],output[i])
    else:
        if isinstance(output, torch.Tensor):
            summary_dict['output_shape'] = list(output.size())
        elif isinstance(output, list):
            summary_dict['output_shape'] = list(np.array(output).shape)
    return summary_dict
 
def summary(input_size, model):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
        
            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            if isinstance(input[0], torch.Tensor):
                summary[m_key]["input_shape"] = list(input[0].size())
            elif isinstance(input[0], list):
                summary[m_key]["input_shape"] = list(np.array(input[0]).shape)

            summary[m_key] = get_output_size(summary[m_key], output)
        
            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            #if hasattr(module, 'bias'):
            #  params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
        
            summary[m_key]['nb_params'] = params
        
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))
    
    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        # x = [Variable(torch.rand(1,*in_size)) for in_size in input_size]
        x = [Variable(torch.rand(*in_size)) for in_size in input_size]
    else:
        # x = Variable(torch.rand(1,*input_size))
        x = Variable(torch.rand(*input_size))

    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()
    
    return summary
#####################################################


def detect(cfg,opt):
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')
    view_img = opt.view  # 实时显示输出结果
    device = select_device(logger, opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        pass
        # shutil.rmtree(opt.save_dir)  # delete dir
    else:
        os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint)
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size

    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.nc_names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    cfg.MODEL.NOT_INIT_MODEL[0] = True  #执行inference
    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out= model(img)
        if not os.path.exists('model_shape1.json'):
            ####获取模型input/output shape#################
            x = summary(img.shape, model)
            info = {}
            for name, age in x.items():
                # print(name, age.items())
                # print('----------------------------------')
                info[name] = str(dict(age.items()))
            with open('model_shape.json', 'w') as f:
                json.dump(info, f)  
            #####################################
        t2 = time_synchronized()
        inf_out, _ = det_out
        inf_time.update(t2-t1,img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))
        det = det_pred[0]

        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        _, _, height, width = img.shape
        h, w, _ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        # ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        # ll_seg_mask = connect_lane(ll_seg_mask)

        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, _, is_demo=True)

        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            xyxy, conf, cls = det[:, :4], det[:, 4:5], det[:, 5:6]            
            if len(xyxy) and len(conf) and len(cls):
                plot_one_box(xyxy, img_det, cls, conf)

        # Stream results
        if view_img:
            cv2.imshow(str(path), img_det)
            cv2.waitKey(1)  # 1 millisecond

        if dataset.mode == 'images':
            cv2.imwrite(save_path, img_det)
        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'C:\Users\haidu\Desktop\YOLOP\weights\model_best.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default=r'C:\Users\haidu\Desktop\YOLOP\inference\images\det_val', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default=r'C:\Users\haidu\Desktop\YOLOP\inference\output\videos', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--view', type=bool, default= False, help='real time show results')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
