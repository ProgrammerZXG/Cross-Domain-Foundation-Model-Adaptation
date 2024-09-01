import logging
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import BasicDataset
from models.adapter import dinov2_mla,dinov2_pup,dinov2_linear
from models.Dpt import dinov2_dpt
from models.unet import U_Net
import numpy as np
from tensorboardX import SummaryWriter
from torchmetrics.classification import JaccardIndex
from loss.FocalLoss import Focal_Loss
from loss.DiceLoss import DiceLoss
from loss.WeightDiceLoss import WeightedDiceLoss
import random
import argparse
# from logger import Logger
import loralib as lora

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def main(args,logger):
    dir_checkpoint = '../checkpoint/' + args.dataset + "/" + args.loss + "/" +args.netType
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    if args.dataset == 'seam':
        args.n1, args.n2 = 1006, 782
        args.classes = 6
        args.patch_h = 72
        args.patch_w = 56
        args.batch_size = 3
    elif args.dataset == 'salt':
        args.n1, args.n2 = 224, 224
        args.classes = 2
        args.patch_h = 20
        args.patch_w = 20
        args.batch_size = 32
    elif args.dataset == 'crater':
        args.n1, args.n2 = 1022, 1022 
        args.classes = 2
        args.patch_h = 73
        args.patch_w = 73
        args.batch_size = 3
    elif args.dataset == 'das':
        args.n1, args.n2 = 512, 512 
        args.classes = 2
        args.patch_h = 37
        args.patch_w = 37
        args.batch_size = 6
    elif args.dataset == 'fault':
        args.n1, args.n2 = 896, 896 
        args.classes = 2
        args.patch_h = 64
        args.patch_w = 64
        args.batch_size = 6

    if args.checkpointName in ["unfrozen","lora"]:
        frozen = False
    elif args.checkpointName == "frozen":
        frozen = True

    if args.netType == "unet":
        net = U_Net(1,args.classes)
    elif args.netType == "linear":
        net = dinov2_linear(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)
    elif args.netType == "mla":
        net = dinov2_mla(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)
    elif args.netType == "pup":
        net = dinov2_pup(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)
    elif args.netType == "dpt":
        net = dinov2_dpt(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)

    logger.info(f'\t{args.netType} NetWork:\n'
                 f'\t{args.classes } num classes\n'
                 f'\t{args.dataset} dataset\n'
                 f'\t{args.vt} vitType\n'
                 f'\t{args.loss} loss\n')
    # net = torch.nn.DataParallel(net, device_ids=range(device_count))
    goTrain(args,
            dir_checkpoint,
            net=net,
            patch_h = args.patch_h,
            patch_w = args.patch_w,
            epochs=args.epochs,
            batch_size= int(args.batch_size),
            learning_rate= args.lr,
            num_classes = args.classes,
            save_checkpoint=args.save_checkpoint
                )
def goTrain(args,
            dir_checkpoint,
            net,
            patch_h,
            patch_w,
            num_classes : int,
            epochs:int = 5,
            batch_size: int = 1,
            learning_rate: float = 1e-4,
            save_checkpoint: bool = True):

    net.to(device)
    get_parameter_number(net)

    # Create dataset
    train_set = BasicDataset(patch_h, patch_w, args.dataset,args.netType, train_mode=True)
    valid_set = BasicDataset(patch_h, patch_w, args.dataset,args.netType, train_mode=False)

    #Create data loaders
    train_loader= DataLoader(dataset = train_set,batch_size = batch_size, shuffle=True)
    valid_loader= DataLoader(dataset = valid_set,batch_size = batch_size, shuffle=False)

    logger.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_set)}
        Validation size: {len(valid_set)}
        Checkpoints:     {save_checkpoint}
    ''')

    jaccard = JaccardIndex(task='multiclass',num_classes=num_classes).to(device)
    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling 
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    # optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.05)
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.01,betas=[0.7,0.999])
    if args.al:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "focal":
        criterion = Focal_Loss(args.classes,device=args.device)
    elif args.loss == "dice":
        criterion = DiceLoss(args.classes)
    elif args.loss == "wdice":
        criterion = WeightedDiceLoss(args.classes,device=args.device)
    elif args.loss == "bace":
        if args.dataset == "seam":
            weight = torch.tensor([1.216,0.395,3.673,0.573,14.193,1.798]).reshape(-1,1).to(args.device)
            criterion = nn.CrossEntropyLoss(weight=weight)

    #Tensorboard open
    writer = SummaryWriter('../Tensorboard/'+args.dataset+'/' + args.loss + '/')

    # Begin training
    train_loss = []
    valid_loss=[]
    train_iou = []
    valid_iou = []
    train_pa = []
    valid_pa = []
    MaxTrainIoU = 0
    MaxValidIoU = 0
    MinTrainLoss = 1e7
    MinValidLoss = 1e7
    net.train()
    warmup_steps = 10
    ini_lr = learning_rate*10
    for epoch in range(1,epochs+1):
        if args.al=="True":
            if epoch < warmup_steps:
                warmup_percent_done = epoch/warmup_steps
                optimizer.param_groups[0]['lr'] = ini_lr * warmup_percent_done
            else:
                scheduler.step()
        total_train_loss = []      
        total_valid_loss = [] 
        total_train_iou = []
        total_valid_iou = []
        total_train_pa = []
        total_valid_pa = []
        with tqdm(total = len(train_set),desc=f'Epoch {epoch}/{epochs}',unit = 'img') as t:
            for data,label in train_loader:
                b1,b2,c,h,w = data.shape
                data = data.to(device).reshape(b1*b2,c,h,w)
                b1,b2,c,h,w = label.shape
                label = label.to(device).reshape(b1*b2,h,w)
                optimizer.zero_grad()
                outputs = net(data,(args.n1,args.n2))
                if args.loss == "bce":
                    loss = criterion(outputs,label.unsqueeze(1).expand(-1, 2, -1, -1).float())
                else:
                    loss = criterion(outputs,label.long())
                _, preds = torch.max(outputs, 1)
                iou_tmp = jaccard(preds,label.long()).detach().cpu().numpy()
                pa_tmp = ((preds == label).sum().item() / (b1*b2*h*w))
                loss.backward()
                optimizer.step()
                t.update(batch_size)
                t.set_postfix(**{'train_loss': loss.item(),'iou': iou_tmp,'accuracy':pa_tmp,'lr': optimizer.param_groups[0]['lr']})
                total_train_loss.append(loss.item())
                total_train_iou.append(iou_tmp)
                total_train_pa.append(pa_tmp)
            train_loss.append(np.mean(total_train_loss))
            train_iou.append(np.mean(total_train_iou))
            train_pa.append(np.mean(total_train_pa))
            logger.info(f"Epoch {epoch} - TrainSet - Loss: {train_loss[-1]}, IoU: {train_iou[-1]}, Accuracy: {train_pa[-1]}")

        # if save_checkpoint and epoch%5==0:
        #     torch.save(net.state_dict(), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_epoch"+str(epoch)+"_train.pth")
        if train_iou[-1]>MaxTrainIoU:
            torch.save(net.state_dict(), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_maxiou_train.pth")
            if args.checkpointName=="lora":
                torch.save(lora.lora_state_dict(net), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_maxiou_train_lora.pth")
            MaxTrainIoU = train_iou[-1]
            logger.info(f'max_train_iou saved!')
        if train_loss[-1]<MinTrainLoss:
            torch.save(net.state_dict(), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_minloss_train.pth")
            if args.checkpointName=="lora":
                torch.save(lora.lora_state_dict(net), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_minloss_train_lora.pth")
            MinTrainLoss = train_loss[-1]
            logger.info(f'min_train_loss saved!')

        net.eval()   
        with tqdm(total = len(valid_set),desc=f'Epoch {epoch}/{epochs}',unit = 'img') as t:                     
            for data,label in valid_loader:
                b1,b2,c,h,w = data.shape
                data = data.to(device).reshape(b1*b2,c,h,w)
                b1,b2,c,h,w = label.shape
                label = label.to(device).reshape(b1*b2,h,w)
                optimizer.zero_grad()
                outputs = net(data,(args.n1,args.n2)) 
                if args.loss == "bce":
                    loss = criterion(outputs,label.unsqueeze(1).expand(-1, 2, -1, -1).float())
                else:
                    loss = criterion(outputs,label.long())
                _, preds = torch.max(outputs, 1)
                iou_tmp = jaccard(preds,label.long()).detach().cpu().numpy()
                pa_tmp = ((preds == label).sum().item() / (b1*b2*h*w))
                optimizer.step()
                t.update(batch_size)
                t.set_postfix(**{'valid_loss': loss.item(),'iou': iou_tmp,'accuracy':pa_tmp,'lr': optimizer.param_groups[0]['lr']})
                total_valid_loss.append(loss.item())
                total_valid_iou.append(iou_tmp)
                total_valid_pa.append(pa_tmp)
            valid_loss.append(np.mean(total_valid_loss))
            valid_iou.append(np.mean(total_valid_iou))	
            valid_pa.append(np.mean(total_valid_pa))	
            logger.info(f"Epoch {epoch} - ValidateSet - Loss: {valid_loss[-1]}, IoU: {valid_iou[-1]}, Accuracy: {valid_pa[-1]}")

        # if save_checkpoint and epoch%5==0:
        #     torch.save(net.state_dict(), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_epoch"+str(epoch)+"_valid.pth")
        if valid_iou[-1]>MaxValidIoU:
            torch.save(net.state_dict(), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_maxiou_valid.pth")
            if args.checkpointName=="lora":
                torch.save(lora.lora_state_dict(net), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_maxiou_valid_lora.pth")
            MaxValidIoU = valid_iou[-1]
            logger.info(f'max_valid_iou saved!') 
        if valid_loss[-1]<MinValidLoss:
            torch.save(net.state_dict(), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_minloss_valid.pth")
            if args.checkpointName=="lora":
                torch.save(lora.lora_state_dict(net), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_minloss_valid_lora.pth")
            MinValidLoss = valid_loss[-1]
            logger.info(f'min_valid_loss saved!') 
        #Tensorboard writting
        writer.add_scalars('loss_' + args.netType + '_' +args.checkpointName + "_" + args.vt,{'train':train_loss[epoch-1],
                                                 'valid':valid_loss[epoch-1]},epoch)
        writer.add_scalars('metrics_' + args.netType + '_' +args.checkpointName + "_" + args.vt,
                           {'train_iou':train_iou[epoch-1],
                            'valid_iou':valid_iou[epoch-1],
                            'train_acc':train_pa[epoch-1],
                            'valid_acc':valid_pa[epoch-1],
                            },epoch)      
        writer.add_scalars('lr_' + args.netType + '_' +args.checkpointName + "_" + args.vt,{'lr':optimizer.param_groups[0]['lr']},epoch)                                                

    #Tensorboard close
    writer.close()

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model Total: %d'%total_num)
    logger.info('Model Trainable: %d'%trainable_num)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Epochs')
    parser.add_argument('--learning_rate', '-l', dest='lr', type=float, default=1e-3,help='Learning rate')
    parser.add_argument('--loss', '-loss',  type=str, default='ce')
    parser.add_argument('--anneal_lr', '-a', dest='al', type=str, default="False")
    parser.add_argument('--dpt', '-p', type=str, default="True", help='dinov2 pretrain')
    parser.add_argument('--vt', '-v', type=str, default="small")
    parser.add_argument('--checkpointName', '-cp',  type=str, default='lora', help='lora or unfrozen')
    parser.add_argument('--netType', '-net',  type=str, default='pup', help='pup,mla,linear,unet')
    parser.add_argument('--dataset', '-d', type=str, default='seam')
    parser.add_argument('--device', '-dn', type=str, default='cuda:4')
    parser.add_argument('--save_checkpoint', '-sckp', type=bool, default=True)
    return parser.parse_args()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    device = args.device
    # logging.info(args.dataset + "/" + args.loss + "/" + args.netType +'/'+ args.checkpointName + "_" + args.vt)
    # log_dir = "../log/" + args.dataset + "/" + args.loss + "/" + args.netType
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # log = Logger(log_dir +'/'+ args.checkpointName + "_" + args.vt+ ".txt")
    # logger = log.getlog()

    log_dir = "../log/" + args.dataset + "/" + args.loss + "/" + args.netType
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{args.checkpointName}_{args.vt}.txt")
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info(f"{args.dataset}/{args.loss}/{args.netType}/{args.checkpointName}_{args.vt}")
    logger = logging.getLogger()
    main(args,logger)
