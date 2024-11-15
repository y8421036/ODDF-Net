from collections import OrderedDict
import random
import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
import model.ODNet_seg_leakyrelu_head2_3_fuse as ODNet
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
import time as time2
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter # 配合老版本torch
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
    model_save_path = os.path.join(opt.saveroot, 'check_points_OD_b4')
    best_model_save_path = os.path.join(opt.saveroot, 'best_model_OD_b4')

    print("Start Setup dataset reader")
    train_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.train_ids,opt.data_size,opt.modality_filename,opt.label_filename,is_dataaug=True) # 数据增强
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True) ##, num_workers=4
    valid_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.val_ids, opt.data_size, opt.modality_filename,opt.label_filename,is_dataaug=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)##

    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)
    Loss_CE = nn.CrossEntropyLoss()
    Loss_Dice = utils.DiceLoss()

    iters = 1 #记录patch总次数
    # disease_lb=np.load('/data2/datasets/OCTA-500/6M/6M-Text labels.npy').astype(np.float32)
    # lb2 = np.zeros(opt.batch_size).astype(np.string_)

    # pred_BGR = np.zeros((opt.data_size[1], opt.data_size[2], 3)).astype(np.uint8)
    # lb_BGR = np.zeros((opt.data_size[1], opt.data_size[2], 3)).astype(np.uint8)
    for epoch in range(1, opt.num_epochs + 1):

        # epoch = epoch+205 #训练中断情况下加载pth继续训练时。保证模型保存序号和评价结果曲线序号能续上
        
        net.train()
        pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader))
        for itr, (train_images, train_annotations, cubename) in pbar:
            # for b in range(opt.batch_size):
            #     name = cubename[b]
            #     name = int(name[:-4])
            #     lb2[b]= np.argwhere(disease_lb==name)[1]  # 根据cubename确定文件名(无扩展名)对应的疾病标签#####
            train_images =train_images.to(device=device, dtype=torch.float32)
            train_annotations = train_annotations.to(device=device, dtype=torch.long)
            patch_times=int(opt.data_size[1]/opt.train_size[1]*opt.data_size[2]/opt.train_size[2])
            for i in range(patch_times):
                train_images_patch, train_annotations_patch = utils.get_patch_random(train_images, train_annotations, opt.data_size, opt.train_size)
                optimizer.zero_grad()
                pred,_ = net(train_images_patch)
                ce = Loss_CE(pred,train_annotations_patch).cuda()
                # dice = Loss_Dice(pred, train_annotations_patch).cuda()

                # loss = 0.6*dice + ce
                loss = ce
                # loss = 0.6*dice + 0.4*ce
                # loss = dice
                # loss = 0.5*dice + 0.5*ce
                
                # writer.add_scalar('loss_total', loss.item(), iters)
                writer.add_scalar('loss_ce', ce.item(), iters)
                # writer.add_scalar('loss_dice', dice.item(), iters)
                iters = iters + 1                

                loss.backward()
                optimizer.step()
        
        # if epoch % 50 == 0:
        #     batch_n = 0
        #     #img_patch = train_images_patch[batch_n].data.cpu().numpy().astype(np.uint8)
        #     lb_tb = train_annotations_patch[batch_n].data.cpu().numpy().astype(np.uint8)
        #     lb_BGR[:,:,0] = np.where(lb_tb==0,53,0) + np.where(lb_tb==1,143,0) + np.where(lb_tb==2,28,0) + np.where(lb_tb==3,186,0) + np.where(lb_tb==4,106,0)
        #     lb_BGR[:,:,1] = np.where(lb_tb==0,32,0) + np.where(lb_tb==1,165,0) + np.where(lb_tb==2,25,0) + np.where(lb_tb==3,131,0) + np.where(lb_tb==4,217,0)
        #     lb_BGR[:,:,2] = np.where(lb_tb==0,15,0) + np.where(lb_tb==1,171,0) + np.where(lb_tb==2,215,0) + np.where(lb_tb==3,43,0) + np.where(lb_tb==4,166,0)
        #     pred_tb = torch.nn.functional.softmax(pred, dim=1)[batch_n].cpu().detach().numpy()
        #     pred_BGR[:,:,0] = 53 * pred_tb[0] + 143 * pred_tb[1] + 28 * pred_tb[2] + 186 * pred_tb[3] + 106 * pred_tb[4]
        #     pred_BGR[:,:,1] = 32 * pred_tb[0] + 165 * pred_tb[1] + 25 * pred_tb[2] + 131 * pred_tb[3] + 217 * pred_tb[4]
        #     pred_BGR[:,:,2] = 15 * pred_tb[0] + 171 * pred_tb[1] + 215 * pred_tb[2] + 43 * pred_tb[3] + 166 * pred_tb[4]
        #     #writer.add_images('image_patch', img_patch, epoch, dataformats='CNHW')
        #     writer.add_image('train_label', lb_BGR, epoch, dataformats='HWC')
        #     writer.add_image('train_pred', pred_BGR, epoch, dataformats='HWC')

        with torch.no_grad():
            torch.save(net.module.state_dict(),os.path.join(model_save_path,f'{epoch}.pth')) #双卡
            # torch.save(net.state_dict(),os.path.join(model_save_path,f'{epoch}.pth')) #单卡
            logging.info(f'Checkpoint {epoch} saved !')
            val_miou_sum = 0
            # val_macc_sum = 0 ############
            val_iou_sum = np.zeros((4,1))
            net.eval()
            pbar = tqdm(enumerate(BackgroundGenerator(valid_loader)), total=len(valid_loader))
            for itr, (test_images, test_annotations,cubename) in pbar:
                result = np.zeros((DATA_SIZE[1], DATA_SIZE[2]))
                test_images = test_images.to(device=device, dtype=torch.float32)
                test_annotations = np.squeeze(test_annotations).cpu().detach().numpy()
                pred,_ = utils.split_test(test_images,net, opt.data_size, opt.train_size, opt.n_classes)
                pred_argmax = torch.argmax(pred, dim=1)
                result[:, :] = pred_argmax[0, 0, :, :].cpu().detach().numpy()

                # 改utils中split_test输出pred2，然后和疾病label做metric
                # acc = utils.cal_acc() #####################
                # val_macc_sum += acc

                miou,iou = utils.cal_miou(result,test_annotations)
                val_miou_sum += miou
                val_iou_sum += iou
            val_miou = val_miou_sum/val_num
            val_iou = val_iou_sum/val_num
            # val_acc = val_macc_sum/val_num
            writer.add_scalar('valid_mIoU', val_miou, epoch)
            writer.add_scalar('valid_iou_F',val_iou[3], epoch)
            writer.add_scalar('valid_iou_V',val_iou[2], epoch)
            writer.add_scalar('valid_iou_A',val_iou[1], epoch)
            writer.add_scalar('valid_iou_C',val_iou[0], epoch)
            # writer.add_scalar('valid_acc_disease',val_acc, epoch)

            if val_miou > best_valid_miou:
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
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    opt = TrainOptions().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = ODNet.ODNet(in_channels=opt.in_channels, n_classes=opt.n_classes)
    #net.apply(model.init_weight)
    net=torch.nn.DataParallel(net,[0,1]).cuda() #双卡
    # net = net.cuda() #单卡
    if opt.load:
        #------------读双卡训出来的pth至双卡net-------------
        new_state_dict = OrderedDict()
        for key,value in torch.load(opt.load, map_location=device).items():
            name = 'module.' + key
            new_state_dict[name] = value
        net.load_state_dict(new_state_dict)
        #------------读双卡训出来的pth至双卡net-------------

        # net.load_state_dict(torch.load(opt.load, map_location=device)) # 读双卡训出来的pth至单卡net
        logging.info(f'Model loaded from {opt.load}')
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




