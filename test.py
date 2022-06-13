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
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--dataset', default='../dataset_ISIC/', type=str)  # cardiac
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
parser.add_argument('--direc', default='./Visualization/Synapse_PFormer_test', type=str, help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=256)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='no', type=str)
parser.add_argument('--loaddirec', default='checkpoints/xxx.pth', type=str)
parser.add_argument('--eval_mode', default='slice', type=str)
parser.add_argument('--visual', default=True, type=bool)

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

tf_train = JointTransform2D(img_size=args.imgsize, crop=crop, p_flip=0, p_rota=0.5, p_scale=0.5,
                            color_jitter_params=None, long_mask=True)  # image reprocessing
tf_val = JointTransform2D(img_size=args.imgsize, crop=crop, p_flip=0, p_gama=0, color_jitter_params=None,
                          long_mask=True)  # image reprocessing
train_dataset = ImageToImage2D(args.dataset, 'train', tf_train,
                               args.classes)  # only random horizontal flip, return image, mask, and filename
val_dataset = ImageToImage2D(args.dataset, 'test', tf_val, args.classes)  # no flip, return image, mask, and filename
# test_dataset = Image2D(args.dataset, 'test')  # no flip, return image and file name
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda")

if args.modelname == "APFormer":
    model = models.P2UtransR.APFormer_Model_Test(n_channels=imgchant, n_classes=args.classes, imgsize=args.imgsize)

# print("Let's use", torch.cuda.device_count(), "GPUs!")
# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
# model = nn.DataParallel(model, device_ids = [0,1]).cuda()

model.to(device)
model.load_state_dict(torch.load(args.loaddirec))
# model.load_state_dict(torch.load(args.loaddirec, map_location=device)['model'])
model.eval()

#for name in model.state_dict():
    #print(name)
# print(model.state_dict()['down4.transformer.layers.1.0.fn.to_qkv.weight'])
# h1 = model.state_dict()['down3.transformer.layers.0.0.fn.to_qkv.weight']
# h2 = model.state_dict()['down3.transformer.layers.1.0.fn.to_qkv.weight']
# h3 = model.state_dict()['down4.transformer.layers.0.0.fn.to_qkv.weight']
# h4 = model.state_dict()['down4.transformer.layers.1.0.fn.to_qkv.weight']

# ortholoss = ortho_loss()
# head_loss = ortholoss(h1, h2, h3, h4)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))
print('Params = ' + str(pytorch_total_params / 1000 ** 2) + 'M')
# flop, param = get_model_complexity_info(model, (1, 256, 256), as_strings=True, print_per_layer_stat=False)
# print("GFLOPs: {}".format(flop))
# print("Params: {}".format(param))
#  ============================================= begin to eval the model =============================================
dices = 0
hds = 0
ious = 0
ses = 0
sps = 0
accs, fs, ps, rs = 0, 0, 0, 0
times = 0
smooth = 1e-25
times = 0
if eval_mode == "slice":
    mdices, mhds = np.zeros(args.classes), np.zeros(args.classes)
    mses, msps, mious = np.zeros(args.classes), np.zeros(args.classes), np.zeros(args.classes)
    maccs, mfs, mps, mrs = np.zeros(args.classes), np.zeros(args.classes), np.zeros(args.classes), np.zeros(
        args.classes)
    for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
        # print(batch_idx)
        if isinstance(rest[0][0], str):
            image_filename = rest[0][0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

        test_img_path = os.path.join(args.dataset + '/img', image_filename)
        from utils.imgname import keep_img_name
        keep_img_name(test_img_path)

        # print(image_filename)
        X_batch = Variable(X_batch.to(device='cuda'))
        y_batch = Variable(y_batch.to(device='cuda'))

        start_time = time.time()
        with torch.no_grad():
            y_out, qkvs1, attns1 = model(X_batch)
        times += time.time() - start_time

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
            # se, sp, iou = metrics.sespiou_coefficient(pred_i, gt_i)
            se, sp, iou, acc, f, precision, recall = metrics.sespiou_coefficient2(pred_i, gt_i)
            maccs[i] += acc
            mfs[i] += f
            mps[i] += precision
            mrs[i] += recall

            mses[i] += se
            msps[i] += sp
            mious[i] += iou
            del pred_i, gt_i
        if args.visual:
            #'''
            img_ori = cv2.imread(os.path.join(args.dataset + '/img', image_filename))

            img = np.zeros((h, w, 3))
            img_r = img_ori[:, :, 0]
            img_g = img_ori[:, :, 1]
            img_b = img_ori[:, :, 2]
            table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135]])
            seg0 = seg[0, :, :]
            # seg0 = gt[0, :, :]

            for i in range(1, args.classes):
                img_r[seg0 == i] = table[i - 1, 0]
                img_g[seg0 == i] = table[i - 1, 1]
                img_b[seg0 == i] = table[i - 1, 2]

            img[:, :, 0] = img_r
            img[:, :, 1] = img_g
            img[:, :, 2] = img_b
            img = np.uint8(img)
            #'''

            #img = np.uint8(seg[0, :, :]*255)

            fulldir = args.direc + "/"
            if not os.path.isdir(fulldir):
                os.makedirs(fulldir)
            cv2.imwrite(fulldir + image_filename, img)

    mdices = mdices / (batch_idx + 1)
    mhds = mhds / (batch_idx + 1)
    mses = mses / (batch_idx + 1)
    msps = msps / (batch_idx + 1)
    mious = mious / (batch_idx + 1)

    maccs = maccs / (batch_idx + 1)
    mfs = mfs / (batch_idx + 1)
    mps = mps / (batch_idx + 1)
    mrs = mrs / (batch_idx + 1)
    for i in range(1, args.classes):
        dices += mdices[i]
        hds += mhds[i]
        ses += mses[i]
        sps += msps[i]
        ious += mious[i]

        accs += maccs[i]
        fs += mfs[i]
        ps += mps[i]
        rs += mrs[i]
    print(mdices, '\n', mhds, '\n', mses, '\n', msps, '\n', mious, '\n')
    print(dices / (args.classes - 1), hds / (args.classes - 1), ses / (args.classes - 1), sps / (args.classes - 1),
          ious / (args.classes - 1))
    print(accs / (args.classes - 1), fs / (args.classes - 1), ps / (args.classes - 1), rs / (args.classes - 1))
    print(times)

elif eval_mode == "patient":
    patientnumber = 200  # record the patients
    flag = np.zeros(patientnumber)  # record the patients
    # print("patient")
    tps, fps = np.zeros((patientnumber, args.classes)), np.zeros((patientnumber, args.classes))
    tns, fns = np.zeros((patientnumber, args.classes)), np.zeros((patientnumber, args.classes))
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
        # print('Time: ', stop - start
        gt = y_batch.detach().cpu().numpy()
        y_out = F.softmax(y_out, dim=1)
        pred = y_out.detach().cpu().numpy()
        seg = np.argmax(pred, axis=1)  # b h w -> b s h w
        b, h, w = seg.shape
        patientid = int(image_filename[:3])
        flag[patientid] = 1
        for i in range(1, args.classes):
            pred_i = np.zeros((b, h, w))
            pred_i[seg == i] = 255
            gt_i = np.zeros((b, h, w))
            gt_i[gt == i] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            tps[patientid, i] += tp
            fps[patientid, i] += fp
            tns[patientid, i] += tn
            fns[patientid, i] += fn
        # ---------------the last patient--------------
    patients = np.sum(flag)
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    dice0 = 2 * tps / (2 * tps + fps + fns + 1e-40)  # p c
    mdices = np.mean(dice0, axis=0)  # c
    print(mdices)
    dices = np.mean(mdices[1:], axis=0)
    print(dices)

else:
    flag = np.zeros(200)
    times = 0
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
            start_time = time.time()
            y_out, qkvs1, attns1 = model(X_batch)
            times += time.time() - start_time
        # stop = timeit.default_timer()
        # print('Time: ', stop - start
        gt = y_batch.detach().cpu().numpy()  # b h w
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
                    # mhds[i] += hausdorff_distance(pred_i[0, :, :, :], gt_i[0, :, :, :], distance="manhattan")
                    mhds[i] = 0  # metric.binary.hd95(pred_i, gt_i)
                    se, sp, iou = metrics.sespiou_coefficient(pred_i, gt_i)
                    mses[i] += se
                    msps[i] += sp
                    mious[i] += iou
                    # print(i, mdices[i], mhds[i], mses[i], msps[i], mious[i])
                    # print(patientid, metrics.dice_coefficient(pred_i, gt_i))
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
        mhds[i] += 0  # hausdorff_distance(pred_i[0, :, :, :], gt_i[0, :, :, :], distance="manhattan")
        se, sp, iou = metrics.sespiou_coefficient(pred_i, gt_i)
        mses[i] += se
        msps[i] += sp
        mious[i] += iou
        print(-1, metrics.dice_coefficient(pred_i, gt_i))
        del pred_i, gt_i
    patients = np.sum(flag)
    mdices, mhds, mses, msps, mious = mdices / patients, mhds / patients, mses / patients, msps / patients, mious / patients
    print(mdices, mhds, mses, msps, mious)
    print(times)
    for i in range(1, args.classes):
        dices += mdices[i]
        hds += mhds[i]
        ious += mious[i]
        ses += mses[i]
        sps += msps[i]
    print(dices / (args.classes - 1), hds / (args.classes - 1), ious / (args.classes - 1), ses / (args.classes - 1),
          sps / (args.classes - 1))

