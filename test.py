import torch
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import cv2
import logging
import sys
import os
import utils.utils as utils
import numpy as np
from options.test_options import TestOptions
import natsort
import utils.BatchDataReader as BatchDataReader
import time
from skimage import transform
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score


def test_net(net,device):
    test_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.test_ids, opt.data_size, opt.modality_filename,opt.label_filename,is_dataaug=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    BGR=np.zeros((opt.data_size[1],opt.data_size[2],3))
    test_num = opt.test_ids[1] - opt.test_ids[0]
    disease_lb=np.load(opt.disease_npy,allow_pickle=True).astype(np.int32)

    with torch.no_grad():
        net.eval()
        pbar = tqdm(enumerate(BackgroundGenerator(test_loader)), total=len(test_loader))
        test_miou_sum = 0
        n = opt.n_classes - 1
        test_iou_sum = np.zeros((n,1))
        test_metrics_sum = np.zeros((n,5,1)) # Dice/HD95/ACC/SE/SP
        pred_list = []  
        label_list = []  
        test_pred2_list = [] #单个输入只得到单个分类预测，没法算metric，因此把整个测试集的分类结果一起算。
        test_label2_list = []

        for itr, (test_images, test_annotations,cubename) in pbar:
            test_images = test_images.to(device=device, dtype=torch.float32)
            pred,pred2 = utils.split_test(test_images,net,opt.data_size, opt.train_size,opt.n_classes)
            pred_argmax = torch.argmax(pred, dim=1)
            pred_argmax = pred_argmax.cpu().detach().numpy()
            
            test_annotations = np.squeeze(test_annotations).cpu().detach().numpy()
            miou,iou = utils.cal_miou(pred_argmax[0,0, :, :],test_annotations)
            test_miou_sum += miou
            test_iou_sum += iou
            
            metrics = np.zeros((5,1)) # Dice/HD95/ACC/SE/SP
            for i in range(int(n)):
                pred_ = np.where(pred_argmax[0,0, :, :]==i+1, 1, 0)
                gt_ = np.where(test_annotations==i+1, 1, 0)
                metrics[0] = utils.cal_Dice(pred_,gt_)
                metrics[1] = utils.cal_hd95(pred_,gt_,(0.009868,0.009868)) # 6M的spacing为0.015mmx0.015mm  3M的spacing为0.009868mmx0.009868mm
                metrics[2] = utils.cal_acc(pred_,gt_)
                metrics[3],metrics[4] = utils.cal_se_sp(pred_,gt_)
                test_metrics_sum[i] += metrics

            path1 = os.path.join(opt.saveroot, 'test_results')
            if not os.path.isdir(path1):
                os.makedirs(path1)
            cv2.imwrite(os.path.join(path1, cubename[0]), pred_argmax[0,0, :, :])
            pred_softmax = torch.nn.functional.softmax(pred, dim=1)
            pred_softmax = pred_softmax.cpu().detach().numpy()
            BGR[:,:,0]=0*pred_softmax[0,0,:,:]+59*pred_softmax[0,1,:,:]+28*pred_softmax[0,2,:,:]+186*pred_softmax[0,3,:,:]+106*pred_softmax[0,4,:,:]
            BGR[:,:,1]=0*pred_softmax[0,0,:,:]+102*pred_softmax[0,1,:,:]+25*pred_softmax[0,2,:,:]+131*pred_softmax[0,3,:,:]+217*pred_softmax[0,4,:,:]
            BGR[:,:,2]=0*pred_softmax[0,0,:,:]+158*pred_softmax[0,1,:,:]+215*pred_softmax[0,2,:,:]+43*pred_softmax[0,3,:,:]+166*pred_softmax[0,4,:,:]
            path2 = os.path.join(opt.saveroot, 'test_visuals')
            if not os.path.isdir(path2):
                os.makedirs(path2)
            cv2.imwrite(os.path.join(path2, cubename[0]), BGR)

            #--for testing disease classication------------
            # result2 = torch.argmax(pred2, dim=1)
            # result2 = result2[0].cpu().detach().numpy()
            # test_pred2_list = np.append(test_pred2_list,result2)
            # name = cubename[0]
            # name = int(name[:-4])
            # lb = np.argwhere(disease_lb==name)
            # lb2_test = disease_lb[lb[0][0],1]
            # lb2_test = np.expand_dims(lb2_test,axis=0)
            # lb2_test = np.expand_dims(lb2_test,axis=0)
            # test_label2_list = np.append(test_label2_list,lb2_test)
            #--for testing disease classication------------

            #--for pr curve--------------------------------
            # pred_list.append(pred_softmax[0,:, 0, :, :]) 
            # gt_binary = []
            # for i in range(int(opt.n_classes)):
            #     gt_binary.append(np.where(test_annotations==i, 1, 0))
            # label_list.append(np.array(gt_binary))
            #--for pr curve--------------------------------
            
        # pr_curve(label_list,pred_list)

        with open(opt.saveroot+"/metrics-seg.txt",'a') as f:
            f.write("Test_mIoU:{}".format(test_miou_sum/test_num))
            f.write("\nTest_IoU_C:{}".format(test_iou_sum[0]/test_num))
            f.write("\nTest_IoU_A:{}".format(test_iou_sum[1]/test_num))
            f.write("\nTest_IoU_V:{}".format(test_iou_sum[2]/test_num))
            f.write("\nTest_IoU_F:{}".format(test_iou_sum[3]/test_num))
            f.write("\n\nTest_Dice_C:{}".format(test_metrics_sum[0][0]/test_num))
            f.write("\nTest_HD95_C:{}".format(test_metrics_sum[0][1]/test_num))
            f.write("\nTest_ACC_C:{}".format(test_metrics_sum[0][2]/test_num))
            f.write("\nTest_SE_C:{}".format(test_metrics_sum[0][3]/test_num))
            f.write("\nTest_SP_C:{}".format(test_metrics_sum[0][4]/test_num))
            f.write("\n\nTest_Dice_A:{}".format(test_metrics_sum[1][0]/test_num))
            f.write("\nTest_HD95_A:{}".format(test_metrics_sum[1][1]/test_num))
            f.write("\nTest_ACC_A:{}".format(test_metrics_sum[1][2]/test_num))
            f.write("\nTest_SE_A:{}".format(test_metrics_sum[1][3]/test_num))
            f.write("\nTest_SP_A:{}".format(test_metrics_sum[1][4]/test_num))
            f.write("\n\nTest_Dice_V:{}".format(test_metrics_sum[2][0]/test_num))
            f.write("\nTest_HD95_V:{}".format(test_metrics_sum[2][1]/test_num))
            f.write("\nTest_ACC_V:{}".format(test_metrics_sum[2][2]/test_num))
            f.write("\nTest_SE_V:{}".format(test_metrics_sum[2][3]/test_num))
            f.write("\nTest_SP_V:{}".format(test_metrics_sum[2][4]/test_num))
            f.write("\n\nTest_Dice_F:{}".format(test_metrics_sum[3][0]/test_num))
            f.write("\nTest_HD95_F:{}".format(test_metrics_sum[3][1]/test_num))
            f.write("\nTest_ACC_F:{}".format(test_metrics_sum[3][2]/test_num))
            f.write("\nTest_SE_F:{}".format(test_metrics_sum[3][3]/test_num))
            f.write("\nTest_SP_F:{}".format(test_metrics_sum[3][4]/test_num))

        # test_cla_acc = metrics.accuracy_score(test_label2_list,test_pred2_list)
        # test_cla_prec_micro = metrics.precision_score(test_label2_list,test_pred2_list, average='micro')
        # test_cla_prec_macro = metrics.precision_score(test_label2_list,test_pred2_list, average='macro')
        # test_cla_reca_micro = metrics.recall_score(test_label2_list,test_pred2_list, average='micro')
        # test_cla_reca_macro = metrics.recall_score(test_label2_list,test_pred2_list, average='macro')
        # test_cla_f1_micro = metrics.f1_score(test_label2_list,test_pred2_list, average='micro')
        # test_cla_f1_macro = metrics.f1_score(test_label2_list,test_pred2_list, average='macro')
        # with open(opt.saveroot+"/metrics-cla.txt",'a') as f:
        #     f.write("\nTest_cla_acc:{}".format(test_cla_acc))
        #     f.write("\nTest_cla_prec_micro:{}".format(test_cla_prec_micro))
        #     f.write("\nTest_cla_prec_macro:{}".format(test_cla_prec_macro))
        #     f.write("\nTest_cla_reca_micro:{}".format(test_cla_reca_micro))
        #     f.write("\nTest_cla_reca_macro:{}".format(test_cla_reca_macro))
        #     f.write("\nTest_cla_f1_micro:{}".format(test_cla_f1_micro))
        #     f.write("\nTest_cla_f1_macro:{}".format(test_cla_f1_macro))


def pr_curve(label_list,pred_list):
    num_class = 5
    score_array = np.array(pred_list) # 50,5,y,x
    score_array = np.transpose(score_array,(1,0,2,3)) 
    score_array = score_array.reshape(num_class,-1)
    label_tensor = np.array(label_list) # 50,5,y,x
    label_tensor = np.transpose(label_tensor,(1,0,2,3))
    label_tensor = label_tensor.reshape(num_class,-1)

    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()
    # for i in range(num_class):
    #     precision_dict[i], recall_dict[i], _ = precision_recall_curve(label_tensor[i, :], score_array[i, :])
    #     average_precision_dict[i] = average_precision_score(label_tensor[i, :], score_array[i, :])
    #     print('class ', i, ' ap=', average_precision_dict[i])
    precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(label_tensor.ravel(), score_array.ravel())
    average_precision_dict["micro"] = average_precision_score(label_tensor.ravel(), score_array.ravel(), average="micro")
    print('ap: {0:0.5f}'.format(average_precision_dict["micro"]))
    npy_save = 'logs/new/6M'
    np.save(os.path.join(npy_save,'pr_prec.npy'), precision_dict["micro"])
    np.save(os.path.join(npy_save,'pr_reca.npy'), recall_dict["micro"])
    np.save(os.path.join(npy_save,'ap.npy'), average_precision_dict["micro"])    

    
if __name__ == '__main__':
    restore_path = 'logs/_pth/new/6M/cla/0.825569/14.pth'
    stage = 'cla'  # seg or cla
    print(stage, restore_path)

    if stage == 'cla':
        import model.ODNet_cla_leakyrelu_head2_4_fuse as ODNet
        # import model.ODNet_cla_leakyrelu_head2_4_fuse_vis as ODNet
    elif stage == 'seg':
        import model.ODNet_seg_leakyrelu_head2_3_fuse as ODNet
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    opt = TestOptions().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = ODNet.ODNet(in_channels=opt.in_channels, n_classes=opt.n_classes)

    # net = torch.nn.DataParallel(net, [0,1]).cuda()  #多卡
    net = net.cuda() # 单卡
    net.load_state_dict(torch.load(restore_path, map_location=device))

    try:
        test_net(net=net,device=device)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
