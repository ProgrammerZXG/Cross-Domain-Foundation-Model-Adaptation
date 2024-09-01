import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Focal_Loss(torch.nn.Module):
    def __init__(self, class_num, device, gamma=2, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num
        self.device =device
        self.alpha = Variable(torch.ones(class_num,1)).to(device)

    def forward(self, predict, target):
        B,C,H,W = predict.shape
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num).permute(0,3,1,2) #获取target的one hot编码
        class_mask = class_mask.contiguous().view(B,C,H,W)
        ids = target.view(-1, 1) 
        # alpha1 = self.get_alpha(target)
        alpha = self.alpha[ids.data.view(-1)] # 注意，这里的alpha是给定的一个list(tensor
                                              #),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
        # 同样，原始ce上增加一个动态权重衰减因子
        # print(alpha.shape)
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    
    def get_alpha(self,target):
        cls = torch.arange(self.class_num).reshape(-1,1).to(self.device)
        counts = torch.bincount(torch.flatten(target))
        cls_num = counts[cls]
        denominator = torch.where(cls_num != 0, cls_num.float(), torch.tensor(1))
        alpha = 1/denominator
        alpha_norm = alpha/alpha.sum()
        return alpha_norm
