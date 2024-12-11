from collections import OrderedDict
import random
import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
import model.ODNet_cla_leakyrelu_head2_4_fuse as ODNet
import utils.utils as utils
import shutil
import natsort
from options.train_options import TrainOptions
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import utils.BatchDataReader as BatchDataReader
import scipy.io as io
import cv2
# from torchsummary import summary
from time import time
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter # 配合老版本torch
from sklearn import metrics
#from init_weights import init_weight


def init_seeds(seed=42):
    # 42, 3407, 114514
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == 42:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# def adjust_lr_poly(optimizer, base_lr, current_epoch, max_epoch, power=0.9):
#     lr = base_lr * (1-float(current_epoch)/max_epoch) ** power
#     optimizer.param_groups[0]['lr'] = lr
#     writer.add_scalar('lr', lr, current_epoch)

def train_net(net,device):
    val_num = opt.val_ids[1] - opt.val_ids[0]
    DATA_SIZE = opt.data_size
    best_valid_miou=0
    model_save_path = os.path.join(opt.saveroot, 'check_points')
    best_model_save_path = os.path.join(opt.saveroot, 'best_model')

    print("Start Setup dataset reader")
    train_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.train_ids,opt.data_size,opt.modality_filename,opt.label_filename,is_dataaug=True) # 数据增强
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    valid_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.val_ids, opt.data_size, opt.modality_filename,opt.label_filename,is_dataaug=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        # optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
        optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, net.parameters()), opt.lr, betas=(0.9, 0.99))  # 导入的参数一开始先不更新
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)
    Loss_CE = nn.CrossEntropyLoss()
    # Loss_Dice = utils.DiceLoss()
    # Loss_focal = utils.FocalLoss()
    # Loss_wce = utils.WeightedCrossEntropyLoss()

    iters = 1 #记录迭代总次数
    disease_lb=np.load(opt.disease_npy,allow_pickle=True).astype(np.int32)
    lb2 = np.zeros(opt.batch_size).astype(np.int64)
    lb2 = np.expand_dims(lb2, axis=1)
    lb2 = np.expand_dims(lb2, axis=1) # shape=(2,1,1)

    # pred_BGR = np.zeros((opt.data_size[1], opt.data_size[2], 3)).astype(np.uint8)
    # lb_BGR = np.zeros((opt.data_size[1], opt.data_size[2], 3)).astype(np.uint8)
    for epoch in range(1, opt.num_epochs + 1):
        if epoch==min_freeze: # 3D部分参数解冻并调小学习率
            optimizer.param_groups[0]['lr'] = 0.0001 ########
            for name,param in net.named_parameters():
                if 'in_conv1' in name:
                    param.requires_grad = True
                    optimizer.add_param_group({'params':param}) # 用SGD注释掉，ADAM则保留。与optimizer对应起来。
                elif 'in_conv2' in name:
                    param.requires_grad = True
                    optimizer.add_param_group({'params':param})
                elif 'FPM1' in name:
                    param.requires_grad = True
                    optimizer.add_param_group({'params':param})
                elif 'FPM2' in name:
                    param.requires_grad = True
                    optimizer.add_param_group({'params':param})
                elif 'FPM3' in name:
                    param.requires_grad = True
                    optimizer.add_param_group({'params':param})
        # elif epoch==min2_freeze: # 2D部分参数也解冻
        #     for name,param in net.named_parameters():
        #         if 'fuse1' in name:
        #             param.requires_grad = True
        #             optimizer.add_param_group({'params':param})
        #         elif 'fuse2' in name:
        #             param.requires_grad = True
        #             optimizer.add_param_group({'params':param})
        #         elif 'fuse3' in name:
        #             param.requires_grad = True
        #             optimizer.add_param_group({'params':param})
        #         elif 'SegNet2D' in name:
        #             param.requires_grad = True
        #             optimizer.add_param_group({'params':param})

        net.train()
        pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader))
        for itr, (train_images, train_annotations, cubename) in pbar:
            for b in range(len(cubename)):
                name = cubename[b]
                name = int(name[:-4])
                lb = np.argwhere(disease_lb==name)
                lb2[b][0][0]= disease_lb[lb[0][0],1] # 根据cubename确定文件名(无扩展名)对应的疾病标签
            train_images =train_images.to(device=device, dtype=torch.float32)
            train_annotations = train_annotations.to(device=device, dtype=torch.long)
            patch_times=int(opt.data_size[1]/opt.train_size[1]*opt.data_size[2]/opt.train_size[2])
            for i in range(patch_times):
                train_images_patch, train_annotations_patch = utils.get_patch_random(train_images, train_annotations, opt.data_size, opt.train_size)
                optimizer.zero_grad()
                pred,pred2= net(train_images_patch) 
                ce = Loss_CE(pred,train_annotations_patch).cuda()
                ce2 = Loss_CE(pred2, torch.from_numpy(lb2[:len(cubename)]).cuda()).cuda()
                loss = ce + 0.2*ce2
         
                writer.add_scalar('loss_total', loss.item(), iters)
                writer.add_scalar('loss_ce', ce.item(), iters)
                # writer.add_scalar('loss_dice', dice.item(), iters)
                writer.add_scalar('loss_ce2', ce2.item(), iters)
                # writer.add_scalar('loss_focal', focal.item(), iters)
                # writer.add_scalar('loss_wce', wce.item(), iters)
                iters = iters + 1                
                loss.backward()
                optimizer.step()
        
        with torch.no_grad():
            torch.save(net.module.state_dict(),os.path.join(model_save_path,f'{epoch}.pth')) #多卡
            # torch.save(net.state_dict(),os.path.join(model_save_path,f'{epoch}.pth')) #单卡
            logging.info(f'Checkpoint {epoch} saved !')
            val_miou_sum = 0
            val_iou_sum = np.zeros((4,1))
            net.eval()
            pbar = tqdm(enumerate(BackgroundGenerator(valid_loader)), total=len(valid_loader))
            for itr, (test_images, test_annotations,cubename) in pbar:
                name = cubename[0]
                name = int(name[:-4])
                lb = np.argwhere(disease_lb==name)
                result = np.zeros((DATA_SIZE[1], DATA_SIZE[2]))
                test_images = test_images.to(device=device, dtype=torch.float32)
                test_annotations = np.squeeze(test_annotations).cpu().detach().numpy()
                pred,pred2 = utils.split_test(test_images,net, opt.data_size, opt.train_size, opt.n_classes)
                pred_argmax = torch.argmax(pred, dim=1)
                result[:, :] = pred_argmax[0, 0, :, :].cpu().detach().numpy()
                miou,iou = utils.cal_miou(result,test_annotations)
                val_miou_sum += miou
                val_iou_sum += iou
                              
            val_miou = val_miou_sum/val_num
            val_iou = val_iou_sum/val_num
            writer.add_scalar('valid_mIoU', val_miou, epoch)
            writer.add_scalar('valid_iou_F',val_iou[3], epoch)
            writer.add_scalar('valid_iou_V',val_iou[2], epoch)
            writer.add_scalar('valid_iou_A',val_iou[1], epoch)
            writer.add_scalar('valid_iou_C',val_iou[0], epoch)

            if epoch>=min_freeze and val_miou>best_valid_miou:
                temp = '{:.6f}'.format(val_miou)
                os.mkdir(os.path.join(best_model_save_path,temp))
                temp2= f'{epoch}.pth'
                shutil.copy(os.path.join(model_save_path,temp2),os.path.join(best_model_save_path,temp,temp2))
                model_names = natsort.natsorted(os.listdir(best_model_save_path))
                if len(model_names) == 4:
                    shutil.rmtree(os.path.join(best_model_save_path,model_names[0]))
                best_valid_miou = val_miou

        torch.cuda.empty_cache() # 清理显存
        # adjust_lr_poly(optimizer, base_lr=opt.lr, current_epoch=epoch, max_epoch=opt.num_epochs, power=0.9)

if __name__ == '__main__':
    min_freeze = 10
    # min2_freeze = 20

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    opt = TrainOptions().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = ODNet.ODNet(in_channels=opt.in_channels, n_classes=opt.n_classes)
    #net.apply(model.init_weight)
    net=torch.nn.DataParallel(net,[0,1]).cuda() #多卡
    # net = net.cuda() #读多卡训seg的pth后，用单卡训cla效果非常差！但用多卡训cla效果正常！原因未知！

    # 加载训练好的seg模型参数
    pretrained_dict = OrderedDict()
    for key,value in torch.load(opt.seg_pth, map_location=device).items():
        name = 'module.' + key
        pretrained_dict[name] = value
    model_dict = net.state_dict()
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict and v.shape==model_dict[k].shape} # 将pretrained_dict中不属于model_dict的key删掉
    model_dict.update(pretrained_dict) #更新当前模型的dict
    net.load_state_dict(model_dict) #加载真正需要的已训练好的dict部分

    # 让导入的参数刚开始不更新
    for name,param in net.named_parameters():
        if 'disease' not in name:
            param.requires_grad = False
        # else:
        #     print(name)

    init_seeds(42)
    writer = SummaryWriter(comment=f'_ODNet')

    try:
        t0 = time()
        train_net(net=net,device=device)
        print("It takes ", (time()-t0)/60/60, "hours.")
        writer.close()
    except KeyboardInterrupt:
        # torch.save(net.state_dict(), 'INTERRUPTED.pth')
        # logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)




