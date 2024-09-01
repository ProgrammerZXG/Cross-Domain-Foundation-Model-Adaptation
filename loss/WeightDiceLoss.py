import torch
import torch.nn.functional as F
from torch.autograd import Variable

class WeightedDiceLoss(torch.nn.Module):
    def __init__(self, class_num,device,smooth=1.):
        super(WeightedDiceLoss, self).__init__()
        self.class_num = class_num
        self.smooth = smooth
        self.device = device

    def forward(self, pred, target):
        B,C,H,W = pred.shape
        weights = self.get_weight(target)

        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, self.class_num).permute(0,3,1,2)
        target = target.contiguous().view(B,C,H,W)
        total_loss = 0
        
        for class_idx in range(self.class_num):
            class_true = target[:, class_idx, ...]
            class_pred = pred[:, class_idx, ...]
            intersection = torch.sum(weights[class_idx] * (class_true * class_pred))
            union = torch.sum(weights[class_idx] * class_true) + torch.sum(weights[class_idx] * class_pred)
            dice_loss = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)        
            total_loss += dice_loss

        return total_loss / self.class_num

    def get_weight(self,target):
        cls = torch.arange(self.class_num).reshape(-1,1).to(self.device)
        counts = torch.bincount(torch.flatten(target))
        cls_num = counts[cls]
        denominator = torch.where(cls_num != 0, cls_num.float(), torch.tensor(1e10))
        alpha = 1/denominator
        alpha_norm = alpha/alpha.sum()
        return alpha_norm