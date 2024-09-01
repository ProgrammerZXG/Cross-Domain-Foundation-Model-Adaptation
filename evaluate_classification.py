import torch
from torch.utils.data import DataLoader
from dataset import BasicDataset
from models.adapter import dinov2_mla, dinov2_pup, dinov2_linear
from models.unet import U_Net
from models.Dpt import dinov2_dpt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from torchmetrics.classification import  JaccardIndex
from demo_classification import get_args
from loss.metric import SegmentationMetric
# from logger import Logger
import logging
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import os

def goPredict(net, device, patch_h, patch_w, mPath, loraPath, datasetname, netType):
    if os.path.exists(mPath):
        if args.checkpointName=="lora":
            net.load_state_dict(torch.load(mPath, map_location=device),strict=False)
            net.load_state_dict(torch.load(loraPath, map_location=device),strict=False)
        else:
            net.load_state_dict(torch.load(mPath, map_location=device),strict=True)
    net.to(device)
    net.eval()
    valid_set = BasicDataset(patch_h, patch_w, datasetname,netType,False)
    valid_loader =DataLoader(dataset = valid_set,batch_size = args.batch_size, shuffle=False)
    data_list = []
    predict = []
    target = []
    pca = []
    jaccard = JaccardIndex(task='multiclass',num_classes=args.classes).to(device)
    iou = []
    pa = []
    f1 = []
    total_conf_matrix = np.zeros((6, 6), dtype=int)
    for data, label in valid_loader:
        b1,b2,c,h,w = data.shape
        data = data.to(device).reshape(b1*b2,c,h,w)
        b1,b2,c,h,w = label.shape
        label = label.to(device).reshape(b1*b2,h,w)
        _, preds = torch.max(net(data,(args.n1,args.n2)), 1)
        iou.append(jaccard(preds,label.long()).detach().cpu().numpy())
        f1.append(f1_score(np.squeeze(preds.detach().cpu().numpy()),
                           np.squeeze(label.long().detach().cpu().numpy())))
        predict.append(np.squeeze(preds.detach().cpu().numpy()).astype(np.int8))
        data_list.append(np.squeeze(data[:,0,:,:].detach().cpu().numpy()).astype(np.float32))
        target.append(np.squeeze(label.detach().cpu().numpy()))
        if args.netType != "unet":
            pca.append(PCA_RGB(args,net,data[0,:,:,:]))

        metric = SegmentationMetric(args.classes)
        for j in range(data.shape[0]):
            metric.CM = metric.addBatch(predict[-1][j], target[-1][j])
            conf_matrix = confusion_matrix(target[-1][j].flatten(), predict[-1][j].flatten(),labels=[0, 1, 2, 3, 4, 5])
            total_conf_matrix += conf_matrix
        pa.append(metric.meanPixelAccuracy())

    class_accuracies = [total_conf_matrix[i, i] / total_conf_matrix[i, :].sum() if total_conf_matrix[i, :].sum() != 0 else 0 for i in range(args.classes)]
    for i in range(args.classes):
        logger.info(f"Class {i}: {class_accuracies[i]:.2%}")

    logger.info("miou: %f"%(np.mean(iou)))
    logger.info("F1: %f"%(np.mean(f1)))
    logger.info("mpa: %f"%(np.mean(pa)))
    # np.array(iou).tofile(netType+"_"+args.checkpointName+"_"+args.dataset+"_iouWithDis.dat")
    # np.array(pa).tofile(netType+"_"+args.checkpointName+"_"+args.dataset+"_paWithDis.dat")
    return data_list, predict, target, pca

def f1_score(y_pred, y_true):
    """
    Calculate the F1 score for a 2D image segmentation task.

    :param y_true: Ground truth labels with shape (height, width).
    :param y_pred: Predicted labels with shape (height, width).
    :return: F1 score.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    true_positive = np.sum(y_true & y_pred)
    false_positive = np.sum(~y_true & y_pred)
    false_negative = np.sum(y_true & ~y_pred)

    # compute precision and recall
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0

    # compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    return f1

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model Total: %d'%total_num)
    logger.info('Model Trainable: %d'%trainable_num)

def PCA_RGB(args, model, image):
    # Ensure the image is a correctly preprocessed PyTorch tensor with shape [1, C, H, W]
    image = image.unsqueeze(0)
    _, _, H, W = image.shape

    features,_ = model.encoder.forward_features(image)
    fea_img = features['x_norm_patchtokens']
    fea_img = fea_img.view(fea_img.size(0),int(H / 14),int(W / 14),384)

    # Flatten the feature map to fit PCA
    H, W = fea_img.shape[1], fea_img.shape[2]
    # print(fea_img.shape)
    features_flattened = fea_img.view(H*W, -1).detach().cpu().numpy()
    # print(features_flattened.shape)
    # Apply PCA to obtain the top three principal components
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features_flattened)

    # Reshape and normalize each principal component
    pcs_images = principal_components.T.reshape((3, H, W))
    pcs_images_normalized = (pcs_images - pcs_images.min(axis=(1,2), keepdims=True)) / \
                            (pcs_images.max(axis=(1,2), keepdims=True) - pcs_images.min(axis=(1,2), keepdims=True))

    # Combine the three principal components into an RGB image
    rgb_image = np.stack((pcs_images_normalized[0], pcs_images_normalized[1], pcs_images_normalized[2]), axis=-1)

    return rgb_image

def plot(img,pngPath=None):
    cmap = ListedColormap(['#d3e0f7', '#a6172d']) 
    fig, ax = plt.subplots(figsize=[10, 10], dpi=360)
    im = ax.imshow(img, cmap=cmap)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if pngPath is not None:
        plt.savefig(pngPath, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':
    args = get_args()
    device = args.device
    # log_dir = "../log/" + args.dataset + "/" + args.loss + "/"
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # log = Logger(log_dir + args.netType +'_'+ args.checkpointName + "_" + args.vt+ ".txt",mode='a')
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
    logger = logging.getLogger()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.info(f"{args.dataset}/{args.loss}/{args.netType}/{args.checkpointName}_{args.vt}")

    losstype = "ce"

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
    elif args.netType == "pup":
        net = dinov2_pup(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)
    elif args.netType == "mla":
        net = dinov2_mla(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)
    elif args.netType == "dpt":
        net = dinov2_dpt(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)
        
    get_parameter_number(net)
    logger.info(args.dataset +'/'+ args.netType +'_'+args.checkpointName + "_" + args.vt)
    model_Path =  '../checkpoint/'+ args.dataset +'/'+ args.loss +'/'+args.netType +'/'+args.checkpointName + "_" + args.vt+'_maxiou_valid.pth'
    lora_Path = '../checkpoint/'+ args.dataset +'/'+ args.loss +'/'+args.netType +'/'+args.checkpointName + "_" + args.vt+'_maxiou_valid_lora.pth'
    pngPath = '../png/'+ args.dataset +'/'+ args.netType +'/'+ args.loss +'/' +args.checkpointName + "_" + args.vt+'/'

    if not os.path.exists(pngPath):
        os.makedirs(pngPath)

    predictPath = os.path.join(pngPath,'prediction')
    if not os.path.exists(predictPath):
        os.makedirs(predictPath)

    patch_h = args.patch_h
    patch_w = args.patch_w

    #paint
    cmin = 0
    cmax = args.classes - 1

    data, predict, target, pca = goPredict(net, device, patch_h, patch_w, model_Path, lora_Path ,args.dataset, args.netType)
    for i in range(len(data)):    
        predict[i][0].tofile(predictPath+'/prediction_'+str(i+1)+'.dat')
        plt.figure(figsize=[10,10])
        plt.imshow(data[i][0], vmin=np.min(data[i][0]), vmax=np.max(data[i][0]), cmap='seismic')
        # plt.axis('off')
        plt.savefig(pngPath + 'sx_' + str(i+1) + '.jpg')
        plt.close()

        plt.figure(figsize=[10,10])
        plt.imshow(predict[i][0], vmin=cmin, vmax=cmax, cmap='jet')
        # plt.axis('off')
        plt.savefig(pngPath + 'pre_' + str(i+1) + '.jpg')
        plt.close()

        plt.figure(figsize=[10,10])
        plt.imshow(target[i][0], vmin=cmin, vmax=cmax, cmap='jet')
        # plt.axis('off')
        plt.savefig(pngPath + 'tar_' + str(i+1) + '.jpg')
        plt.close()

        # plot(target[i][0],pngPath + 'label_' + str(i+1) + '.png')

        if args.netType != "unet":
            plt.figure(figsize=(10, 10))
            plt.imshow(pca[i])
            plt.axis('off')
            plt.show()
            plt.savefig(pngPath + 'pca_' + str(i+1) + '.jpg')
            plt.close()