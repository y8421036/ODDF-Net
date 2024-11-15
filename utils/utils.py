import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import SimpleITK as sitk
import surface_distance as surfdist
from sklearn.metrics import confusion_matrix

def check_dir_exist(dir):
    """create directories"""
    if os.path.exists(dir):
        return
    else:
        names = os.path.split(dir)
        dir = ''
        for name in names:
            dir = os.path.join(dir,name)
            if not os.path.exists(dir):
                try:
                    os.mkdir(dir)
                except:
                    pass
        print('dir','\''+dir+'\'','is created.')

def cal_Dice(img1,img2):
    shape = img1.shape
    I = 0
    U = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] >= 1 and img2[i,j] >= 1:
                I += 1
            if img1[i,j] >= 1 or img2[i,j] >= 1:
                U += 1
    return 2*I/(I+U+1e-5)

def cal_hd95(img_pred,img_gt,sapce):
    surface_distances = surfdist.compute_surface_distances(img_gt.astype(np.bool8), img_pred.astype(np.bool8), spacing_mm=sapce)  #注意不同数据集的像素间距不同！改最后的参数！单位是毫米
    hd = surfdist.compute_robust_hausdorff(surface_distances, 95)
    return hd

def cal_acc(img1,img2):
    shape = img1.shape
    acc = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] == img2[i,j]:
                acc += 1
    return acc/(shape[0]*shape[1])

def cal_se_sp(img_pred,img_gt):
    cm = confusion_matrix(img_gt.flatten(), img_pred.flatten())
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    return sensitivity,specificity

def cal_miou(img1,img2):
    classnum = img2.max()
    iou=np.zeros((int(classnum),1))
    for i in range(int(classnum)):
        imga=img1==i+1
        imgb=img2==i+1
        imgi=imga * imgb
        imgu=imga + imgb
        iou[i]=np.sum(imgi)/np.sum(imgu)
    miou=np.mean(iou)
    return miou, iou


def make_one_hot(input, shape):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    result = torch.zeros(shape)
    result.scatter_(1, input.cpu(), 1)
    return result


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)
    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights
def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
    (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class FocalLoss_classify(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5), 
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num=7, alpha=None, gamma=2, size_average=True):
        super(FocalLoss_classify, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1).cpu()
        target = target.contiguous().view(target.shape[0], -1).cpu()

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        shape=predict.shape
        target = torch.unsqueeze(target, 1)
        target=make_one_hot(target.long(),shape)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]

def get_patch_random(data,label, cube_size, patch_size):
    patch_pos = []
    for i in range(3):
        patch_pos.append(torch.randint(0, cube_size[i] - patch_size[i] + 1, (1,)))
        #print(cube_size[i], patch_size[i], patch_pos[i])
    data_crop = data[:, :, patch_pos[0]:patch_pos[0] + patch_size[0], patch_pos[1]:patch_pos[1] + patch_size[1],patch_pos[2]:patch_pos[2] + patch_size[2]]
    data_crop = data_crop.contiguous()
    label_crop = label[:, :, patch_pos[1]:patch_pos[1] + patch_size[1],patch_pos[2]:patch_pos[2] + patch_size[2]]
    label_crop  = label_crop.contiguous()
    return data_crop,label_crop

def split_test(data,model, cube_size, patch_size,n_classes):
    outshape=[1,n_classes,1,cube_size[1],cube_size[2]]
    result = torch.zeros(outshape)
    result = result.to(data.device)
    for x in range(0, cube_size[0], patch_size[0]):
        for y in range(0, cube_size[1], patch_size[1]):
            for z in range(0, cube_size[2], patch_size[2]):
                input=data[:,:,x:x+patch_size[0],y:y+patch_size[1],z:z+patch_size[2]]
                output, pred2=model(input)
                # output, pred2, att1,att2,att3,att4,att5=model(input)
                result[:,:,x:x+patch_size[0],y:y+patch_size[1],z:z+patch_size[2]]=output
    # np.save('logs/att/att1',att1.cpu())
    # np.save('logs/att/att2',att2.cpu())
    # np.save('logs/att/att3',att3.cpu())
    # np.save('logs/att/att4',att4.cpu())
    # np.save('logs/att/att5',att5.cpu())
    return result,pred2

def standardization(data):
    # z-score
    _mean = data.mean()
    _std = data.std()
    return ((data-_mean)/_std).astype(np.float32)

def normalization(data):
    # min-max
    _min = np.min(data)
    _range = np.max(data)-_min
    return ((data-_min)/_range).astype(np.float32)