import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse
import re
import sys
import logging
import os
from albumentations import (
    Compose,
    Normalize,
    Resize,
    PadIfNeeded,
    VerticalFlip,
    HorizontalFlip,
    RandomCrop,
    CenterCrop,
)
from utils import ramps, losses
import ignite.engine as engine
import ignite.handlers as handlers
import ignite.contrib.handlers as c_handlers
import ignite.metrics as imetrics
# visualization
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import (
    OutputHandler,
    OptimizerParamsHandler,
    WeightsScalarHandler,
)
import torchvision.utils as tvutils
import torch.nn.functional as F
# modules
from loss import LossMulti
from dataset import SutureSegDataset, TwoStreamBatchSampler
from metrics import (
    MetricRecord,
    calculate_confusion_matrix_from_arrays,
    calculate_iou,
    calculate_dice,
)
from Models.plane_model import UNet11_DO
import random
os.environ['TORCH_HOME'] = './pretrained_model'
np.seterr(all='raise')


def main(args):
    # check cuda available
    assert torch.cuda.is_available() == True
    # when the input dimension doesnot change, add this flag to speed up
    cudnn.benchmark = True

    # model ckpt name prefix
    model_save_dir = '_'.join([args.model, str(args.lr)])
    # we can add more params for comparison in future experiments
    model_save_dir = '_'.join([model_save_dir, str(args.jaccard_weight), \
                               str(args.batch_size), str(args.input_height), str(args.input_width)])


    # model save directory
    model_save_dir = Path(args.model_save_dir) / model_save_dir
    model_save_dir.mkdir(exist_ok=True, parents=True)
    # model_save_dir.mkdir( parents=True)
    args.model_save_dir = str(model_save_dir)  # e.g. $ROOT_DIR/model/UNet_binary_1e-5/

    # logger
    logging_logger = logging.getLogger('train')
    logging_logger.propagate = False  # this logger should not propagate to parent logger
    # logger log_level
    logging_logger.setLevel(args.log_level)
    # logging format
    formatter = logging.Formatter("[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s")

    # console log handler: write log to console
    rf_handler = logging.StreamHandler()
    rf_handler.setFormatter(formatter)
    logging_logger.addHandler(rf_handler)

    # file log handler: write log to file for each fold
    f_handler = logging.FileHandler(str(model_save_dir / (args.log_filename + '.log')))
    f_handler.setFormatter(formatter)
    logging_logger.addHandler(f_handler)

    # add as args
    args.logging_logger = logging_logger

    if args.tb_log:

        # tensorboard logger
        tf_log_dir = model_save_dir / 'tb_logs'
        tf_log_dir.mkdir(exist_ok=True, parents=True)

        tb_logger = TensorboardLogger(log_dir=str(tf_log_dir))
        # add as arguments
        args.tb_logger = tb_logger
    # input params
    input_msg = 'Input arguments:\n'
    for arg_name, arg_val in vars(args).items():
        input_msg += '{}: {}\n'.format(arg_name, arg_val)
    logging_logger.info(input_msg)

    # metrics mean and std: RESUME HERE
    mean_metrics = {'miou': 0, 'std_miou': 0, 'mdice': 0, 'std_mdice': 0}
    args.mean_metrics = mean_metrics
    # iou_list = []
    # dice_list = []

    train_fold(args)

    avg_results_log = 'average on validation:\n'
    for metric_name, val in mean_metrics.items():
        avg_results_log += '%s: %.5f\n' % (metric_name, val)
    logging_logger.info(avg_results_log)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train_fold(args):
    # loggers
    logging_logger = args.logging_logger
    if args.tb_log:
        tb_logger = args.tb_logger

    num_classes = 2
    # init model
    def create_model(ema=False):
        # Network definition
        model = UNet11_DO(in_channels=3, num_classes=num_classes, bn=True, has_dropout=False)
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    # transform for train/valid data
    train_transform, valid_transform = get_transform(args.model)

    # loss function
    loss_func = LossMulti(num_classes, args.jaccard_weight)


    # DataLoader and Dataset args
    train_shuffle = True  # set shuffle False for lstm
    train_ds_kwargs = {
        'root': args.root,
        'transform': train_transform,
        'mode': 'train',
        'batch_size': args.batch_size,
    }

    valid_num_workers = args.num_workers
    valid_batch_size = args.batch_size

    # additional valid dataset kws
    valid_ds_kwargs = {
        'root': args.root,
        'transform': valid_transform,
        'mode': 'valid',
    }


    labeled_idxs = list(range(1, 61))
    unlabeled_idxs = list(range(61, 541))
    del unlabeled_idxs[::2]
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    # train dataloader
    train_loader = DataLoader(
        dataset=SutureSegDataset(**train_ds_kwargs),
        num_workers=0,
        batch_sampler=batch_sampler,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    # valid dataloader
    valid_loader = DataLoader(
        dataset=SutureSegDataset(**valid_ds_kwargs),
        shuffle=False,  # in validation, no need to shuffle
        num_workers=valid_num_workers,
        batch_size=1,  # in valid time. have to use one image by one
        pin_memory=True
    )

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    ema_model.train()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    consistency_criterion = losses.softmax_mse_loss
    model.train()
    iter_num = 0
    max_iterations = args.max_iterations
    max_epoch = max_iterations // len(train_loader) + 1

    for epoch_num in tqdm(range(max_epoch), ncols=70):

        for i_batch, sampled_batch in enumerate(train_loader):

            # additional params to feed into model
            add_params = {}
            inputs = sampled_batch['input'].cuda(non_blocking=True)
            unlabeled_batch = inputs[args.labeled_bs:]

            targets = sampled_batch['target'].cuda(non_blocking=True)
            names = sampled_batch['name']
            # print(names[:args.labeled_bs])

            noise = torch.clamp(torch.randn_like(unlabeled_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_batch + noise

            outputs = model(inputs, **add_params)

            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
            # T = 8
            # batch_r = unlabeled_batch.repeat(2, 1, 1, 1)
            # stride = batch_r.shape[0] // 2
            # preds = torch.zeros([stride * T, 2, args.input_height, args.input_width]).cuda()
            # for i in range(T // 2):
            #     ema_inputs = batch_r + torch.clamp(torch.randn_like(batch_r) * 0.1, -0.2, 0.2)
            #     with torch.no_grad():
            #         preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
            # preds = F.softmax(preds, dim=1)
            # preds = preds.reshape(T, stride, 2, args.input_height, args.input_width)
            # preds = torch.mean(preds, dim=0)  # (batch, 2, 512, 640)
            # uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)  # (batch, 1, 512, 640)


            supervised_loss = loss_func(outputs[:args.labeled_bs], targets[:args.labeled_bs])
            # loss_seg = F.cross_entropy(outputs[:args.labeled_bs], targets[:args.labeled_bs])
            # outputs_soft = F.softmax(outputs, dim=1)
            # loss_seg_dice = losses.dice_loss(outputs_soft[:args.labeled_bs, 1, :, :], targets[:args.labeled_bs] == 1)
            # supervised_loss = 0.5 * (loss_seg + loss_seg_dice)
            consistency_weight = get_current_consistency_weight(iter_num // (max_iterations/ args.consistency_rampup))

            consistency_dist = consistency_criterion(outputs[args.labeled_bs:], ema_output) #[8, 2, 512, 640]
            consistency_dist = torch.mean(consistency_dist)
            # threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)

            # mask = (uncertainty < threshold).float()
            # consistency_dist = torch.sum( mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
            consistency_loss = consistency_weight * consistency_dist

            loss = supervised_loss+consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            iter_num = iter_num + 1
            logging_logger.info('iteration %d : loss : %f sup_loss : %f cons_loss: %f, loss_weight: %f' % (iter_num, loss.item(), supervised_loss.item(), consistency_loss.item(), consistency_weight))

            ## change lr
            if iter_num % 500 == 0:
                lr_ = args.lr * 0.1 ** (iter_num // 500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 500 == 0:
                save_mode_path = os.path.join(args.model_save_dir, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging_logger.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break

            if iter_num % 50 == 0:
                iou_total = 0
                dice_total = 0
                # record current batch metrics
                iou_mRecords = MetricRecord()
                dice_mRecords = MetricRecord()
                for i_iter, batch in enumerate(valid_loader):
                    with torch.no_grad():
                        model.eval()
                        inputs = batch['input'].cuda(non_blocking=True)
                        targets = batch['target'].cuda(non_blocking=True)

                        # additional arguments
                        add_params = {}

                        # output logits
                        outputs = model(inputs, **add_params)
                        # loss
                        loss = loss_func(outputs, targets)

                        output_softmaxs = torch.softmax(outputs, dim=1)
                        output_argmaxs = output_softmaxs.argmax(dim=1)
                        # output_classes and target_classes: <b, h, w>
                        output_classes = output_argmaxs.cpu().numpy()
                        target_classes = targets.cpu().numpy()


                        cm_b = np.zeros((num_classes, num_classes), dtype=np.uint32)

                        for output_class, target_class in zip(output_classes, target_classes):
                            # calculate metrics for each frame
                            # calculate using confusion matrix or dirctly using definition
                            cm = calculate_confusion_matrix_from_arrays(output_class, target_class, num_classes)
                            iou_mRecords.update_record(calculate_iou(cm))
                            dice_mRecords.update_record(calculate_dice(cm))
                            cm_b += cm
                miou = iou_mRecords.data_mean()
                mdice = dice_mRecords.data_mean()
                print('****************************************************************************************')
                print(' data(%d) mean IoU: %.3f, std: %.3f' % ( len(miou['items']), miou['mean'], miou['std']))
                print(' data(%d) mean Dice: %.3f, std: %.3f' % ( len(mdice['items']), mdice['mean'], mdice['std']))
                print('****************************************************************************************')

    save_mode_path = os.path.join(args.model_save_dir, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging_logger.info("save model to {}".format(save_mode_path))




def get_transform(model_name):
    if 'TAPNet' in model_name:
        # transform for sequences of images is very tricky
        # TODO: more transforms should be adopted for better results
        train_transform_ops = [
            PadIfNeeded(min_height=args.input_height, min_width=args.input_width, p=1),
            Normalize(p=1),
            # optional transforms
            Resize(height=args.input_height, width=args.input_width, p=1),
            # CenterCrop(height=args.input_height, width=args.input_width, p=1)
        ]
    else:
        train_transform_ops = [
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            PadIfNeeded(min_height=args.input_height, min_width=args.input_width, p=1),
            # Normalize(p=1),
            # optional transforms
            # Resize(height=args.input_height, width=args.input_width, p=1),
            # CenterCrop(height=args.input_height, width=args.input_width, p=1)
            RandomCrop(height=args.input_height, width=args.input_width, p=1),
        ]

    valid_transform_ops = [
        # Normalize(p=1),
        PadIfNeeded(min_height=480, min_width=640, p=1),
        # optional transforms
        Resize(height=480, width=640, p=1),
        # CenterCrop(height=args.input_height, width=args.input_width, p=1)
    ]
    return Compose(train_transform_ops, p=1, ), Compose(valid_transform_ops, p=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model training')

    # training settings


    parser.add_argument('--resume', type=bool, default=False,
                        help='whether to resume training.')

    # hyper-params
    parser.add_argument('--model', type=str, default='UNet11_DO',
                        help='model for segmentation.')

    parser.add_argument('--mf', type=bool, default=True,
                        help='whether to enable Motion Flow for attention map generation. This option is only valid for TAPNet based models. Enable this option to use MF-TAPNet')

    parser.add_argument('--max_epochs', type=int, default=20,
                        help='max epochs for training.')

    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate.')

    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='learning rate decay.')

    parser.add_argument('--lr_decay_epochs', type=int, default=5,
                        help='number of epochs for every learning rate decay.')

    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay.')

    parser.add_argument('--es_patience', type=int, default=20,
                        help='early stop patience')

    parser.add_argument('--batch_size', type=int, default=6,
                        help='input batch size.')

    parser.add_argument('--labeled_bs', type=int, default=3, help='labeled_batch_size per gpu')

    parser.add_argument('--jaccard_weight', type=float, default=0.3,
                        help='jaccard weight [0.0, 1.0] in loss function.')

    parser.add_argument('--input_height', type=int, default=256,
                        help='input image height.')

    parser.add_argument('--input_width', type=int, default=320,
                        help='input image width.')


    # dirs
    parser.add_argument('--root', type=str, default='./bg_new_split/split1/',
                        help='train data directory.')

    parser.add_argument('--model_save_dir', type=str, default='./model_ckpt/KEMOVE_test/',
                        help='model save dir.')

    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='model checkpoint directory.')

    # pytorch settings
    parser.add_argument('--device_ids', type=int, default=0, nargs='+',
                        help='GPU devices ids.')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of subprocesses in pytorch DataLoader. 0 means loading data in main process.')

    # log
    parser.add_argument('--tb_log', type=bool, default=False, help='whether to use TensorboardLogger for logging internal results')

    parser.add_argument('--log_level', type=int, default=logging.DEBUG,
                        help='console logging level in python logging module.')

    parser.add_argument('--log_filename', type=str, default='train',
                        help='output log file name')

    # consistency
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
    parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
    parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
    parser.add_argument('--seed', type=int, default=200, help='random seed')
    args = parser.parse_args()
    main(args)
