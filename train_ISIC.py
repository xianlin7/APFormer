import torch
import models
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import cv2
from torch.utils.tensorboard import SummaryWriter
import utils.metrics as metrics
from hausdorff import hausdorff_distance
import time
import torch.nn.functional as F
import random
from utils.loss_functions.dice_loss import DC_and_BCE_loss, DC_and_CE_loss
from utils.loss_functions.atten_matrix_loss import selfsupervise_loss
from einops import rearrange

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#  ============================== add the seed to make sure the results are reproducible ==============================

seed_value = 5000  # the number of seed
np.random.seed(seed_value)  # set random seed for numpy
random.seed(seed_value)  # set random seed for python
os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
torch.manual_seed(seed_value)  # set random seed for CPU
torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
torch.backends.cudnn.deterministic = True  # set random seed for convolution

#  ================================================ parameters setting ================================================

parser = argparse.ArgumentParser(description='Medical Transformer')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--dataset', default='../../dataset_ISIC/', type=str)  # cardiac
parser.add_argument('--modelname', default='APFormer', type=str,
                    help='type of model')
parser.add_argument('--classes', type=int, default=2,
                    help='number of classes')
parser.add_argument('--cuda', default="on", type=str,
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--direc', default='./medt', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=256)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='no', type=str)
parser.add_argument('--tensorboard', default='./tensorboard/', type=str)
parser.add_argument('--eval_mode', default='slice', type=str)
parser.add_argument('--load_path', default='checkpoints_ISIC/xxxx.pth', type=str)
parser.add_argument('--pre_trained', default=False, type=bool)

#  =============================================== model initialization ===============================================

args = parser.parse_args()
direc = args.direc  # the path of saving model
eval_mode = args.eval_mode

if args.gray == "yes":
    from utils.utils_gray import JointTransform2D, ImageToImage2D, Image2D

    imgchant = 1
else:
    from utils.utils_rgb import JointTransform2D, ImageToImage2D, Image2D

    imgchant = 3

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_train = JointTransform2D(img_size=args.imgsize, crop=crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0, p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
tf_val = JointTransform2D(img_size=args.imgsize, crop=crop, p_flip=0, p_gama=0, color_jitter_params=None, long_mask=True)  # image reprocessing
train_dataset = ImageToImage2D(args.dataset, 'train', tf_train, args.classes)  # only random horizontal flip, return image, mask, and filename
val_dataset = ImageToImage2D(args.dataset, 'val', tf_val, args.classes)  # no flip, return image, mask, and filename
test_dataset = ImageToImage2D(args.dataset, 'test', tf_val, args.classes)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
device = torch.device("cuda")

device = torch.device("cuda")

if args.modelname == "APFormer":
    model = models.P2UtransR.APFormer_Model(n_channels=imgchant, n_classes=args.classes, imgsize=args.imgsize)

model.to(device)
if args.pre_trained:
    model.load_state_dict(torch.load(args.load_path))

criterion = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
ssa_loss = selfsupervise_loss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate, weight_decay=1e-5)

'''
trans_params = list(map(id, model.trans4.transformer.parameters()))
base_params = filter(lambda p: id(p) not in trans_params, model.parameters())
params = [{'params': base_params},
          {'params': model.trans4.transformer.parameters(), 'lr': args.learning_rate*0.1}]
optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=1e-5)
'''

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

timestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
boardpath = './tensorboard_ISIC/' + args.modelname + '_' + timestr
if not os.path.isdir(boardpath):
    os.makedirs(boardpath)
TensorWriter = SummaryWriter(boardpath)


#  ============================================= begin to train the model =============================================

best_dice = 0.0
for epoch in range(args.epochs):
    #  ---------------------------------- training ----------------------------------
    model.train()
    train_losses = 0
    bcdice_losses = 0
    head_losses = 0
    attn_losses = 0
    for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):
        X_batch = Variable(X_batch.to(device='cuda'))
        y_batch = Variable(y_batch.to(device='cuda'))

        # ------------------------- forward ------------------------------

        output, sx, attns1 = model(X_batch)
        bcdice_loss = criterion(output, y_batch)  # contain softmax
        attn_loss = ssa_loss(attns1)
        
        y_batch_pool = rearrange(y_batch, 'b (h n) (w m) -> b h w (n m)', n=8, m=8)
        y_batch_pool = torch.sum(y_batch_pool, dim=-1)  # b h w
        y_batch_pool[y_batch_pool > 0] = 1
        head_loss = criterion(sx, y_batch_pool)

        #train_loss = 0.9 * bcdice_loss + 0.1 * attn_loss
        #train_loss = bcdice_loss
        train_loss = 0.8 * bcdice_loss + 0.1 * attn_loss + 0.1*head_loss
        print("train_loss:", train_loss)

        # ------------------------- backward -----------------------------

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_losses += train_loss.item()
        bcdice_losses += bcdice_loss.item()
        attn_losses += attn_loss.item()

    #  ---------------------------- log the train progress ----------------------------
    print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, args.epochs, train_losses / (batch_idx + 1)))
    TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
    TensorWriter.add_scalar('train_bcdice_loss', bcdice_losses / (batch_idx + 1), epoch)
    TensorWriter.add_scalar('train_head_loss', head_losses / (batch_idx + 1), epoch)
    TensorWriter.add_scalar('train_attn_loss', attn_losses / (batch_idx + 1), epoch)


    #  ----------------------------------- evaluate -----------------------------------
    model.eval()
    val_losses = 0
    val_bcdice_losses = 0
    val_head_losses = 0
    val_attn_losses = 0
    dices = 0
    hds = 0
    smooth = 1e-25
    mdices, mhds = np.zeros(args.classes), np.zeros(args.classes)
    if eval_mode == "slice":
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
            X_batch = Variable(X_batch.to(device='cuda'))
            y_batch = Variable(y_batch.to(device='cuda'))
            # start = timeit.default_timer()
            with torch.no_grad():
                y_out, qkvs1, attns1 = model(X_batch)
            # stop = timeit.default_timer()
            # print('Time: ', stop - start)

            val_bcdice_loss = criterion(y_out, y_batch)
            val_losses += val_bcdice_loss.item() 
            
            #val_head_losses += val_head_loss.item()
            val_bcdice_losses += val_bcdice_loss.item()
            #val_attn_losses += val_attn_loss.item()

            gt = y_batch.detach().cpu().numpy()
            y_out = F.softmax(y_out, dim=1)
            pred = y_out.detach().cpu().numpy()  # (b, c,h, w) tep
            seg = np.argmax(pred, axis=1)  # (b, h, w)
            b, h, w = seg.shape
            for i in range(1, args.classes):
                pred_i = np.zeros((b, h, w))
                pred_i[seg == i] = 255
                gt_i = np.zeros((b, h, w))
                gt_i[gt == i] = 255
                mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
                mhds[i] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
                del pred_i, gt_i
        mdices = mdices / (batch_idx + 1)
        mhds = mhds / (batch_idx + 1)
        for i in range(1, args.classes):
            dices += mdices[i]
            hds += mhds[i]

        print(dices / (args.classes - 1), hds / (args.classes - 1))
        print('epoch [{}/{}], test loss:{:.4f}'.format(epoch, args.epochs, val_losses / (batch_idx + 1)))
        print('epoch [{}/{}], test dice:{:.4f}'.format(epoch, args.epochs, dices / (args.classes - 1)))
        TensorWriter.add_scalar('val_loss', val_losses / (batch_idx + 1), epoch)
        TensorWriter.add_scalar('val_bcdice_loss', val_bcdice_losses / (batch_idx + 1), epoch)
        #TensorWriter.add_scalar('val_head_loss', val_head_losses / (batch_idx + 1), epoch)
        TensorWriter.add_scalar('val_attn_loss', val_attn_losses / (batch_idx + 1), epoch)
        TensorWriter.add_scalar('dices', dices / (args.classes - 1), epoch)
        TensorWriter.add_scalar('hausdorff', hds / (args.classes - 1), epoch)

        if epoch == 70:
            for param in model.parameters():
                param.requires_grad = True
        if dices / (args.classes - 1) > best_dice:
            best_dice = dices / ((args.classes - 1))
            timestr = time.strftime('%m%d%H%M')
            save_path = './checkpoints_ISIC/' + args.modelname + '_%s' % timestr + '_' + str(best_dice)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
    else:
        flag = np.zeros(200)  # record the patients
        # print("patient")
        mdices, mhds = np.zeros(args.classes), np.zeros(args.classes)
        mses, msps, mious = np.zeros(args.classes), np.zeros(args.classes), np.zeros(args.classes)
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
            X_batch = Variable(X_batch.to(device='cuda'))
            y_batch = Variable(y_batch.to(device='cuda'))
            # start = timeit.default_timer()
            with torch.no_grad():
                y_out, qkvs1, attns1 = model(X_batch)
            # stop = timeit.default_timer()
            # print('Time: ', stop - start)

            val_bcdice_loss = criterion(y_out, y_batch)

            val_losses += val_bcdice_loss.item()
            
            #val_head_losses += val_head_loss.item()
            val_bcdice_losses += val_bcdice_loss.item()

            gt = y_batch.detach().cpu().numpy()
            y_out = F.softmax(y_out, dim=1)
            pred = y_out.detach().cpu().numpy()
            seg = np.argmax(pred, axis=1)  # b h w -> b s h w
            patientid = int(image_filename[:3])
            if flag[patientid] == 0:
                if np.sum(flag) > 0:  # compute the former result
                    b, s, h, w = seg_patient.shape
                    for i in range(1, args.classes):
                        pred_i = np.zeros((b, s, h, w))
                        pred_i[seg_patient == i] = 1
                        gt_i = np.zeros((b, s, h, w))
                        gt_i[gt_patient == i] = 1
                        mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
                        del pred_i, gt_i
                seg_patient = seg[:, None, :, :]
                gt_patient = gt[:, None, :, :]
                flag[patientid] = 1
            else:
                seg_patient = np.concatenate((seg_patient, seg[:, None, :, :]), axis=1)
                gt_patient = np.concatenate((gt_patient, gt[:, None, :, :]), axis=1)
        # ---------------the last patient--------------
        b, s, h, w = seg_patient.shape
        for i in range(1, args.classes):
            pred_i = np.zeros((b, s, h, w))
            pred_i[seg_patient == i] = 1
            gt_i = np.zeros((b, s, h, w))
            gt_i[gt_patient == i] = 1
            mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
            del pred_i, gt_i
        patients = np.sum(flag)
        mdices = mdices / patients
        for i in range(1, args.classes):
            dices += mdices[i]
        print('epoch [{}/{}], test loss:{:.4f}'.format(epoch, args.epochs, val_losses / (batch_idx + 1)))
        TensorWriter.add_scalar('val_loss', val_losses / (batch_idx + 1), epoch)
        TensorWriter.add_scalar('dices', dices / (args.classes - 1), epoch)
        if epoch == 70:
            for param in model.parameters():
                param.requires_grad = True
        if dices / (args.classes - 1) > best_dice or epoch == args.epochs - 1:
            best_dice = dices / (args.classes - 1)
            timestr = time.strftime('%m%d%H%M')
            save_path = './checkpoints_ISIC/' + args.modelname + '_%s' % timestr + '_' + str(best_dice)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)