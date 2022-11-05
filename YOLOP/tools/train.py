import argparse
import os, sys
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from lib.utils import DataLoaderX, torch_distributed_zero_first
from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    # general
    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)

    # philly
    parser.add_argument('--modelDir',help='model directory',type=str,default='')
    parser.add_argument('--logDir',help='log directory',type=str,default='runs/')
    parser.add_argument('--dataDir',help='data directory',type=str,default='')
    parser.add_argument('--prevModelDir',help='prev Model directory',type=str,default='')

    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()

    return args


def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)

    # Set DDP variables
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    rank = global_rank
   
    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'train', rank=rank)

    if rank in [-1, 0]:
        logger.info(pprint.pformat(args))
        #logger.info(cfg)

        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
    else:
        writer_dict = None

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # bulid up model
    # start_time = time.time()
    print("begin to bulid up model...")
    # DP mode
    device = select_device(logger, device=str(cfg.GPUS[0]), batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU* len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')

    # device = select_device(logger=, 'cpu')

    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    
    print("load model to device")
    model = get_net(cfg).to(device)
    print("finish build model")
    
    # define loss function (criterion) and optimizer
    criterion = get_loss(cfg, device=device)
    optimizer = get_optimizer(cfg, model)

    Encoder_para_idx = [str(i) for i in range(0, 17)]
    Det_Head_para_idx = [str(i) for i in range(17, 25)]
    Da_Seg_Head_para_idx = [str(i) for i in range(25, 34)]
    Ll_Seg_Head_para_idx = [str(i) for i in range(34,43)]

    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    #加载模型
    if rank in [-1, 0]:
        checkpoint_file = os.path.join(
            os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET), 'checkpoint.pth'
        )
        if os.path.exists(cfg.MODEL.PRETRAINED):
            logger.info("=> loading model '{}'".format(cfg.MODEL.PRETRAINED))
            checkpoint = torch.load(cfg.MODEL.PRETRAINED)
            begin_epoch = checkpoint['epoch']
            
            #修改detection部分
            list_key_dect =["model.24.anchors", "model.24.anchor_grid", "model.24.m.0.weight", "model.24.m.0.bias",
            "model.24.m.1.weight", "model.24.m.1.bias", "model.24.m.2.weight", "model.24.m.2.bias"]   
            checkpoint['state_dict'][list_key_dect[2]].resize_(39, 128, 1 , 1)
            checkpoint['state_dict'][list_key_dect[3]].resize_(39)
            checkpoint['state_dict'][list_key_dect[4]].resize_(39, 256, 1 , 1)
            checkpoint['state_dict'][list_key_dect[5]].resize_(39)
            checkpoint['state_dict'][list_key_dect[6]].resize_(39, 512, 1 , 1)
            checkpoint['state_dict'][list_key_dect[7]].resize_(39)
            model.load_state_dict(checkpoint['state_dict'])
            #加载预训练完检测后的模型
            checkpoint1 = torch.load(r'E:\CnnModel\YOLOP\tools\runs\BddDataset\_2021-11-15-10-50\model_best.pth')
            model.load_state_dict(checkpoint1)
            
            #修改optimizer部分
            checkpoint['optimizer']['state'][185]['exp_avg'].resize_(39, 128, 1 , 1)
            checkpoint['optimizer']['state'][185]['exp_avg']= torch.zeros_like(torch.empty(39, 128, 1 , 1))
            checkpoint['optimizer']['state'][185]['exp_avg_sq'].resize_(39, 128, 1 , 1)
            checkpoint['optimizer']['state'][185]['exp_avg_sq'] = torch.zeros_like(torch.empty(39, 128, 1 , 1))
            checkpoint['optimizer']['state'][186]['exp_avg'].resize_(39)
            checkpoint['optimizer']['state'][186]['exp_avg'] = torch.zeros_like(torch.empty(39))
            checkpoint['optimizer']['state'][186]['exp_avg_sq'].resize_(39)
            checkpoint['optimizer']['state'][186]['exp_avg_sq'] = torch.zeros_like(torch.empty(39))
            checkpoint['optimizer']['state'][187]['exp_avg'].resize_(39, 256, 1 , 1)
            checkpoint['optimizer']['state'][187]['exp_avg'] = torch.zeros_like(torch.empty(39, 256, 1 , 1))
            checkpoint['optimizer']['state'][187]['exp_avg_sq'].resize_(39, 256, 1 , 1)
            checkpoint['optimizer']['state'][187]['exp_avg_sq'] = torch.zeros_like(torch.empty(39, 256, 1 , 1))
            checkpoint['optimizer']['state'][188]['exp_avg'].resize_(39)
            checkpoint['optimizer']['state'][188]['exp_avg'] = torch.zeros_like(torch.empty(39))
            checkpoint['optimizer']['state'][188]['exp_avg_sq'].resize_(39)
            checkpoint['optimizer']['state'][188]['exp_avg_sq'] = torch.zeros_like(torch.empty(39))
            checkpoint['optimizer']['state'][189]['exp_avg'].resize_(39, 512, 1 , 1)
            checkpoint['optimizer']['state'][189]['exp_avg'] = torch.zeros_like(torch.empty(39, 512, 1 , 1))
            checkpoint['optimizer']['state'][189]['exp_avg_sq'].resize_(39, 512, 1 , 1)
            checkpoint['optimizer']['state'][189]['exp_avg_sq'] = torch.zeros_like(torch.empty(39, 512, 1 , 1))
            checkpoint['optimizer']['state'][190]['exp_avg'].resize_(39)
            checkpoint['optimizer']['state'][190]['exp_avg'] = torch.zeros_like(torch.empty(39))
            checkpoint['optimizer']['state'][190]['exp_avg_sq'].resize_(39)
            checkpoint['optimizer']['state'][190]['exp_avg_sq'] = torch.zeros_like(torch.empty(39))
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.MODEL.PRETRAINED, checkpoint['epoch']))
        
        if os.path.exists(cfg.MODEL.PRETRAINED_DET):
            logger.info("=> loading model weight in det branch from '{}'".format(cfg.MODEL.PRETRAINED))
            det_idx_range = [str(i) for i in range(0,25)]
            model_dict = model.state_dict()
            checkpoint_file = cfg.MODEL.PRETRAINED_DET
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            last_epoch = checkpoint['epoch']
            checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict)
            logger.info("=> loaded det branch checkpoint '{}' ".format(checkpoint_file))
        
        if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

        if cfg.TRAIN.SEG_ONLY:  #Only train two segmentation branchs
            logger.info('freeze encoder and Det head...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.DET_ONLY:  #Only train detection branch
            logger.info('freeze encoder and two Seg heads...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Da_Seg_Head_para_idx + Ll_Seg_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.ENC_SEG_ONLY:  # Only train encoder and two segmentation branchs
            logger.info('freeze Det head...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers 
                if k.split(".")[1] in Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.ENC_DET_ONLY:    # Only train encoder and detection branchs
            logger.info('freeze two Seg heads...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Da_Seg_Head_para_idx + Ll_Seg_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.LANE_ONLY: 
            logger.info('freeze encoder and Det head and Da_Seg heads...')
            # print(model.named_parameters)
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Da_Seg_Head_para_idx + Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.DRIVABLE_ONLY:
            logger.info('freeze encoder and Det head and Ll_Seg heads...')
            # print(model.named_parameters)
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Ll_Seg_Head_para_idx + Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

    #多个GPU并行训练    
    # if rank == -1 and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
        # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
        
    # DDP mode
    # if rank != -1:
    #     model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)


    # assign model params
    model.gr = 1.0 #
    model.nc = cfg.num_dect_class
    # print('bulid model finished')

    # Data loading
    print("begin to load data")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(   # 相当于调用dataset.BddDataset(...)
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None

    train_loader = DataLoaderX( #把数据集从磁盘读到内存中
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle= cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        sampler=train_sampler,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    num_batch = len(train_loader)

    if rank in [-1, 0]:
        valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            cfg=cfg,
            is_train=False,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        valid_loader = DataLoaderX(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=dataset.AutoDriveDataset.collate_fn
        )
        print('load data finished')
    
    if rank in [-1, 0]:
        if cfg.NEED_AUTOANCHOR:
            logger.info("begin check anchors")
            run_anchor(logger,train_dataset, model=model, thr=cfg.TRAIN.ANCHOR_THRESHOLD, imgsz=min(cfg.MODEL.IMAGE_SIZE))
        else:
            logger.info("anchors loaded successfully")
            det = model.module.model[model.module.detector_index] if is_parallel(model) \
                else model.model[model.detector_index]
            # logger.info(str(det.anchors))

    # training
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    print('=> start training...')
    best_loss = 1000
    for epoch in range(begin_epoch+1, cfg.TRAIN.END_EPOCH+1):
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)
       
        train(cfg, train_loader, model, criterion, optimizer, scaler,epoch, 
            num_batch, num_warmup, logger, final_output_dir, device, rank)
   
        lr_scheduler.step()
       
        if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH) and rank in [-1, 0]:
            da_segment_results,ll_segment_results,detect_results, total_loss, maps, times = validate(
                epoch, cfg, valid_loader, model, criterion, final_output_dir, device )

            msg = 'Epoch: [{0}]    Loss({loss:.3f})\n' \
                'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame) plot({t_plot:.4f}s/frame)'.format(
                    epoch, loss=total_loss, da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1],
                    da_seg_miou=da_segment_results[2], ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1],
                    ll_seg_miou=ll_segment_results[2], p=detect_results[0], r=detect_results[1], map50=detect_results[2],
                    map=detect_results[3], t_inf=times[0], t_nms=times[1], t_plot=times[2])
            logger.info(msg)

        # save checkpoint model and best model
        if rank in [-1, 0]:
            savepath = os.path.join(final_output_dir, f'epoch-{epoch}.pth')
            logger.info('=> saving checkpoint to {}'.format(savepath))
            if total_loss < best_loss:
                best_loss = total_loss
                is_best = True
                save_checkpoint(epoch=epoch, name=cfg.MODEL.NAME, model=model, optimizer=optimizer,
                output_dir=final_output_dir, filename= None, is_best = is_best)
            save_checkpoint(epoch=epoch, name=cfg.MODEL.NAME, model=model, optimizer=optimizer,
            output_dir=os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET), filename=f'epoch-{epoch}.pth')

    # save final model
    if rank in [-1, 0]:
        final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        logger.info('=> saving final model state to {}'.format(final_model_state_file))
        model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
        torch.save(model_state, final_model_state_file)
        writer_dict['writer'].close()
    else:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()