import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, class_num, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
    def forward(self, pred, target):
        B,C,H,W = pred.shape
        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, self.class_num).permute(0,3,1,2)
        target = target.contiguous().view(B,C,H,W)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        return loss