import time
from lib.core.evaluate import ConfusionMatrix,SegmentationMetric
from lib.core.general import non_max_suppression,check_img_size,scale_coords,xyxy2xywh,xywh2xyxy,box_iou,coco80_to_coco91_class,plot_images,ap_per_class,output_to_target
from lib.utils.utils import time_synchronized
from lib.utils import plot_one_box,show_seg_result
from lib.config import cfg
from lib.dataset.convert import id_dict, id_dict_single
import torch
import numpy as np
from pathlib import Path
import cv2
import os
import math
from torch.cuda import amp
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def train(cfg, train_loader, model, criterion, optimizer, scaler, epoch, num_batch, num_warmup, logger, output_dir, device, rank=-1):
    losses = AverageMeter()

    #新添加可视化训练结果
    save_dir = output_dir + os.path.sep + 'visualization_train'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cfg.MODEL.NOT_INIT_MODEL[0] = True  #执行inference

    # switch to train mode
    model.train()
    ran_batch = np.random.randint(0, len(train_loader))
    for batch_i, (input, target, paths, shapes) in enumerate(train_loader):
        num_iter = batch_i + num_batch * (epoch - 1)
        start = time.time()
        if num_iter < num_warmup:
            # warm up
            lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                           (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
            xi = [0, num_warmup]
            for j, x in enumerate(optimizer.param_groups):   
                x['lr'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])

        if not cfg.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            _, _, height, width = input.shape    #batch size, channel, height, width
            
        with amp.autocast(enabled=device.type != 'cpu'):
            if cfg.TRAIN.PLOT:
                detect_out_tr, road_seg_out_tr, lane_seg_out_tr = model(input)
                dect_inf_out, dect_train_out = detect_out_tr
                total_loss, head_losses = criterion((dect_train_out,road_seg_out_tr, lane_seg_out_tr), target, shapes, model)
                del road_seg_out_tr, lane_seg_out_tr, dect_train_out
                torch.cuda.empty_cache()   
            else:
                outputs = model(input)
                total_loss, head_losses = criterion(outputs, target, shapes,model)

        # compute gradient and do update step
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if rank in [-1, 0]:
            # measure accuracy and record loss
            losses.update(total_loss.item(), input.size(0))
            end = time.time()
            if batch_i % cfg.PRINT_FREQ == 0:
                msg = 'Epoch:[{}][{}/{}] Len_Time:{:0.5f}s ' \
                      'l_box:{:0.5f} l_obj:{:0.5f} l_cls:{:0.5f} l_road:{:0.5f} l_lane:{:0.5f} l_lane_iou:{:0.5f}' \
                      ' Loss:{:0.5f}'.format( epoch, batch_i, len(train_loader), end-start,\
                          head_losses[0],head_losses[1],head_losses[2],\
                          head_losses[3],head_losses[4],head_losses[5],head_losses[6])
                logger.info(msg)

        with torch.no_grad():
            if cfg.TRAIN.PLOT:
                ### 可视化训练结果是否可靠：
                target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
                NMS_output = non_max_suppression(dect_inf_out, conf_thres= cfg.TEST.NMS_CONF_THRESHOLD, iou_thres=cfg.TEST.NMS_IOU_THRESHOLD, labels=[])
                if batch_i == ran_batch:
                # if True:
                    for i in range(len(paths)):
                        img_name = os.path.basename(paths[i]).split('.')[0]
                        # print(img_name)
                        img_det = input[i]                    
                        img_det = img_det.cpu().detach().numpy()
                        img_det = img_det.transpose(1, 2, 0)
                        # while(1):                          
                        #     img = cv2.imread(paths[i])
                        #     cv2.imshow("org_img", img)
                        #     cv2.imshow("aug_img", img_det)
                        #     if cv2.waitKey(1)&0XFF==27:
                        #         break
                        # cv2.destroyAllWindows()  
                        img_gt = img_det.copy()
                        det = NMS_output[i].clone()
                        if len(det):
                            det[:, :4] = scale_coords(input[i].shape[1:], det[:,:4], img_det.shape).round()
                        xyxy, conf, cls = det[:, :4], det[:, 4:5], det[:, 5:6]      
                        plot_one_box(xyxy, img_det, cls, conf)
                        cv2.imwrite(save_dir+"/{}_{}_{}_det_pred.png".format(epoch,batch_i, img_name), img_det)

                        ### targht:(batch_idex,cls,x,y,w,h) 正确的标签
                        labels = target[0][target[0][:, 0] == i, 1:]
                        labels[:, 1:5]=xywh2xyxy(labels[:, 1:5])
                        if len(labels):
                            labels[:,1:5]=scale_coords(input[i].shape[1:], labels[:,1:5], img_gt.shape).round()
                        cls = labels[:, 0]
                        xyxy = labels[:,1:5]
                        gt_scores = np.ones([len(cls)])
                        plot_one_box(xyxy, img_gt, cls, gt_scores)
                        cv2.imwrite(save_dir+"/{}_{}_{}_det_gt.png".format(epoch,batch_i, img_name), img_gt)
                        del img_det, img_gt, det, xyxy, conf, cls, labels, gt_scores
                        
                        # del img_det, img_gt, det, xyxy, conf, cls
                        torch.cuda.empty_cache()


def validate(epoch, config, val_loader, model, criterion, output_dir, device='cpu'):
    # setting
    cfg.MODEL.NOT_INIT_MODEL[0] = True  #执行inference
    max_stride = 32
    save_dir = output_dir + os.path.sep + 'visualization_val'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # print(save_dir)
    img_stride = max([check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE]) #img_stride满足最大步长max_stride的倍数
    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(config.GPUS)
    training = False
    save_conf=False # save auto-label confidences
    verbose=False #信息,冗长的
    save_hybrid=False
    nc = model.nc  # 检测的类别数
    iouv = torch.linspace(0.5, 0.95, 10).to(device)     # iou vector for mAP@0.5:0.95
    niou = iouv.numel() #返回数组中元素的个数

    seen =  0 
    confusion_matrix = ConfusionMatrix(nc)                 # detector confusion matrix
    road_metric = SegmentationMetric(config.num_seg_class)   # road_segment confusion matrix    
    lane_metric = SegmentationMetric(config.num_seg_class)                      # lane_segment confusion matrix

    #返回检测类别的索引
    # nc_names = {k: v for k, v in enumerate(model.nc_names if hasattr(model, 'nc_names') else model.module.nc_names)}
    nc_names = id_dict
    
    p, r, mp, mr, map50, map70, map75, map, t_inf, t_nms , t_plot= 0., 0., 0., 0., 0., 0., 0., 0., 0. , 0., 0.
    
    losses = AverageMeter()
    road_acc_seg = AverageMeter()
    road_IoU_seg = AverageMeter()
    road_mIoU_seg = AverageMeter()
    lane_acc_seg = AverageMeter()
    lane_IoU_seg = AverageMeter()
    lane_mIoU_seg = AverageMeter()

    T_inf = AverageMeter()
    T_nms = AverageMeter()
    T_plot = AverageMeter()

    # switch to val mode
    model.eval()
    stats, ap, ap_class = [], [], []
 
    #shapes = (h0, w0), ((h / h0, w / w0), pad) 

    ran_batch = np.random.randint(0, len(val_loader))
    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader)):
        if not config.DEBUG:
            img = img.to(device, non_blocking=True) #non_blocking用于计算与数据传输的并行
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            nb, _, height, width = img.shape    #batch size, channel, height, width

        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]  # 图像填充的宽高
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = shapes[0][1][0][0] # 1280/640 = 0.5
            #前向推理时间
            t = time_synchronized()
            tplot = time_synchronized()
            detect_out, road_seg_out, lane_seg_out= model(img)  # 模型预测结果
            t_inf = time_synchronized() - t
            if batch_i > 0:
                T_inf.update(t_inf/img.size(0), img.size(0))

            dect_inf_out, dect_train_out = detect_out

            #road area segment evaluation
            _, da_predict = torch.max(road_seg_out, 1)
            _, da_gt = torch.max(target[1], 1)
            da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            road_metric.reset()
            road_metric.addBatch(da_predict.cpu(), da_gt.cpu())
            da_acc = road_metric.pixelAccuracy()
            da_IoU = road_metric.IntersectionOverUnion()
            da_mIoU = road_metric.meanIntersectionOverUnion()

            road_acc_seg.update(da_acc, img.size(0))
            road_IoU_seg.update(da_IoU, img.size(0))
            road_mIoU_seg.update(da_mIoU, img.size(0))

            #lane line segmentation evaluation
            _,ll_predict=torch.max(lane_seg_out, 1)
            _,ll_gt=torch.max(target[2], 1)
            ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            ll_gt = ll_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            lane_metric.reset()
            lane_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
            ll_acc = lane_metric.lineAccuracy()
            ll_IoU = lane_metric.IntersectionOverUnion()
            ll_mIoU = lane_metric.meanIntersectionOverUnion()

            lane_acc_seg.update(ll_acc, img.size(0))
            lane_IoU_seg.update(ll_IoU, img.size(0))
            lane_mIoU_seg.update(ll_mIoU, img.size(0))
            
            total_loss, head_losses = criterion((dect_train_out,road_seg_out, lane_seg_out), target, shapes, model)   #Compute loss
            losses.update(total_loss.item(), img.size(0))

            #NMS
            t = time_synchronized()
            target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device) 
            lb = [target[0][target[0][:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            ## targht: (batch_idex,cls,x,y,w,h) width=640, height=384
            NMS_output = non_max_suppression(dect_inf_out, conf_thres= config.TEST.NMS_CONF_THRESHOLD, iou_thres=config.TEST.NMS_IOU_THRESHOLD, labels=lb)
            ## NMS_output:(x1, y1, x2, y2, conf, cls)
            
            t_nms = time_synchronized() - t
            if batch_i > 0:
                T_nms.update(t_nms/img.size(0),img.size(0))

            if config.TEST.PLOTS:
                if batch_i == ran_batch:
                # if True:
                    for i in range(len(paths)):
                        img_name = os.path.basename(paths[i]).split('.')[0]
                        #绘制时间
                        img_test = cv2.imread(paths[i])
                        da_seg_mask = road_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        da_seg_mask = torch.nn.functional.interpolate(da_seg_mask, scale_factor=int(1/ratio), mode='bilinear', align_corners=True)
                        _, da_seg_mask = torch.max(da_seg_mask, 1)

                        da_gt_mask = target[1][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        da_gt_mask = torch.nn.functional.interpolate(da_gt_mask, scale_factor=int(1/ratio), mode='bilinear', align_corners=True)
                        _, da_gt_mask = torch.max(da_gt_mask, 1)

                        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
                        da_gt_mask = da_gt_mask.int().squeeze().cpu().numpy()
                        
                        img_test1 = img_test.copy()
                        _ = show_seg_result(img_test, da_seg_mask, batch_i, i ,epoch, save_dir)
                        _ = show_seg_result(img_test1, da_gt_mask, batch_i, i , epoch, save_dir, is_gt=True)

                        img_ll = cv2.imread(paths[i])
                        ll_seg_mask = lane_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_mask, scale_factor=int(1/ratio), mode='bilinear', align_corners=True)
                        _, ll_seg_mask = torch.max(ll_seg_mask, 1)

                        ll_gt_mask = target[2][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_gt_mask = torch.nn.functional.interpolate(ll_gt_mask, scale_factor=int(1/ratio), mode='bilinear', align_corners=True)
                        _, ll_gt_mask = torch.max(ll_gt_mask, 1)

                        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
                        ll_gt_mask = ll_gt_mask.int().squeeze().cpu().numpy()
                       
                        img_ll1 = img_ll.copy()
                        _ = show_seg_result(img_ll, ll_seg_mask, batch_i, i ,epoch,save_dir, is_ll=True)
                        _ = show_seg_result(img_ll1, ll_gt_mask, batch_i, i ,epoch, save_dir, is_ll=True, is_gt=True)

                        img_det = cv2.imread(paths[i])
                        img_gt = img_det.copy()
                        det = NMS_output[i].clone()
                        if len(det):
                            det[:, :4] = scale_coords(img[i].shape[1:], det[:,:4], img_det.shape).round()
                            xyxy, conf, cls = det[:, :4], det[:, 4:5], det[:, 5:6]      
                            plot_one_box(xyxy, img_det, cls, conf)
                            cv2.imwrite(save_dir+"/{}_{}_{}_det_pred.png".format(epoch,batch_i,img_name), img_det)
                        t_plot = time_synchronized() - tplot
                        if batch_i > 0:
                            T_plot.update(t_plot/img.size(0), img.size(0))

                        ### targht:(batch_idex,cls,x,y,w,h) 正确的标签

                        # labels = target[0][target[0][:, 0] == i, 1:]
                        # labels[:, 1:5]=xywh2xyxy(labels[:, 1:5])
                        # if len(labels):
                        #     labels[:,1:5]=scale_coords(img[i].shape[1:], labels[:,1:5], img_gt.shape).round()
                        #     cls = labels[:, 0]
                        #     xyxy = labels[:,1:5]
                        #     gt_scores = np.ones([len(cls)])
                        #     plot_one_box(xyxy, img_gt, cls, gt_scores)
                        #     cv2.imwrite(save_dir+"/{}_{}_{}_det_gt.png".format(epoch,batch_i,img_name), img_gt)


        # Statistics per image
        for si, pred in enumerate(NMS_output):
            labels = target[0][target[0][:, 0] == si, 1:]     #all object in one image 
            nl = len(labels)    # num of object
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

             # Append to text file
            if config.TEST.SAVE_TXT:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    if not os.path.exists(os.path.join(save_dir,'labels')):
                        os.mkdir(os.path.join(save_dir,'labels'))
                    with open(os.path.join(save_dir,'labels', path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                # if config.TEST.PLOTS:
                if True:
                    confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):                    
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # n*m  n:pred  m:label
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))      

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy  zip(*) :unzip

    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names = nc_names)
        ap50, ap70, ap75,ap = ap[:, 0], ap[:,4], ap[:,5],ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map70, map75, map = p.mean(), r.mean(), ap50.mean(), ap70.mean(),ap75.mean(),ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%15s'+'%12s' * 8  # print format
    print(pf % ('all', 'num_test', 'nt_sum', 'mp', 'mr', 'map50', 'map70', 'map75', 'map'))
    pf = '%15s' + '%12.3g' * 8  # print format
    print(pf % ('average', seen, nt.sum(), mp, mr, map50, map70, map75, map))

    # Print results per class
    if (verbose or (nc <= 20 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (nc_names[c], seen, nt[c], p[i], r[i], ap50[i], ap70[i], ap75[i], ap[i]))

    # Plots
    # if config.TEST.PLOTS:
    if True:
        confusion_matrix.plot(save_dir=save_dir, names=list(nc_names.values()))

    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    da_segment_result = (road_acc_seg.avg,road_IoU_seg.avg,road_mIoU_seg.avg)
    ll_segment_result = (lane_acc_seg.avg,lane_IoU_seg.avg,lane_mIoU_seg.avg)

    detect_result = np.asarray([mp, mr, map50, map])
    
    #print segmet_result
    t = [T_inf.avg, T_nms.avg, T_plot.avg]
    return da_segment_result, ll_segment_result, detect_result, losses.avg, maps, t
