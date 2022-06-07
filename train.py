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
    valid_batch_size = 1

    # additional valid dataset kws
    valid_ds_kwargs = {
        'root': args.root,
        'transform': valid_transform,
        'mode': 'valid',
    }





    # train dataloader
    train_loader = DataLoader(
        dataset=SutureSegDataset(**train_ds_kwargs),
        shuffle=train_shuffle,
        num_workers=args.num_workers,
        pin_memory=True,

    )
    # valid dataloader
    valid_loader = DataLoader(
        dataset=SutureSegDataset(**valid_ds_kwargs),
        shuffle=False,  # in validation, no need to shuffle
        num_workers=valid_num_workers,
        batch_size=valid_batch_size,  # in valid time. have to use one image by one
        pin_memory=True
    )

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def train_step(engine, batch):
        model.train()

        optimizer.zero_grad()

        # additional params to feed into model
        add_params = {}
        inputs = batch['input'].cuda(non_blocking=True)
        with torch.no_grad():
            targets = batch['target'].cuda(non_blocking=True)
        outputs = model(inputs, **add_params)

        loss_kwargs = {}

        loss = loss_func(outputs, targets, **loss_kwargs)
        loss.backward()
        optimizer.step()

        return_dict = {
            'output': outputs,
            'target': targets,
            'loss_kwargs': loss_kwargs,
            'loss': loss.item(),
        }

        return return_dict
    # init trainer
    trainer = engine.Engine(train_step)
    step_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                               step_size=args.lr_decay_epochs, gamma=args.lr_decay)
    lr_scheduler = c_handlers.param_scheduler.LRScheduler(step_scheduler)
    trainer.add_event_handler(engine.Events.EPOCH_STARTED, lr_scheduler)

    @trainer.on(engine.Events.STARTED)
    def trainer_start_callback(engine):

        # resume training
        if args.resume:
            # ckpt for current fold fold_<fold>_model_<epoch>.pth
            ckpt_dir = Path(args.ckpt_dir)
            ckpt_filename = str(ckpt_dir) + ('/fold_%d_iter_6000.pth' % fold)
            # ckpt_filename = str(ckpt_dir) + ('/fold_%d_model_46.pth' % fold)
            # res = re.match(r'fold_%d_model_(\d+).pth' % fold, ckpt_filename)
            # res = str(ckpt_filename).split('_')[-1].split('.')[0]
            # restore epoch
            engine.state.epoch = int(19)
            # load model state dict
            model.load_state_dict(
                torch.load('../model_ckpt/MT/UNet11_binary_0.0001_0.3_16_512_640/fold_0_iter_6000.pth'), strict=False)
            logging_logger.info('restore model [{}] from epoch {}.'.format(args.model, engine.state.epoch))
        else:
            logging_logger.info('train model [{}] from scratch'.format(args.model))

        # record metrics history every epoch
        engine.state.metrics_records = {}

    @trainer.on(engine.Events.EPOCH_STARTED)
    def trainer_epoch_start_callback(engine):
        # log learning rate on pbar
        train_pbar.log_message('model: %s,,, lr: %.5f, batch size: %d' % \
                               (args.model, lr_scheduler.get_param(), args.batch_size))

    @trainer.on(engine.Events.ITERATION_COMPLETED)
    def trainer_iter_comp_callback(engine):
        # logging_logger.info(engine.state.metrics)
        pass

    # monitor loss
    # running average loss
    train_ra_loss = imetrics.RunningAverage(output_transform=lambda x: x['loss'], alpha=0.98)
    train_ra_loss.attach(trainer, 'train_ra_loss')

    # monitor train loss over epoch
    train_loss = imetrics.Loss(loss_func, output_transform=lambda x: (x['output'], x['target']))
    train_loss.attach(trainer, 'train_loss')

    # progress bar
    train_pbar = c_handlers.ProgressBar(persist=True, dynamic_ncols=True)
    train_metric_names = ['train_ra_loss']
    train_pbar.attach(trainer, metric_names=train_metric_names)



    def valid_step(engine, batch):
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

            # record current batch metrics
            iou_mRecords = MetricRecord()
            dice_mRecords = MetricRecord()

            cm_b = np.zeros((num_classes, num_classes), dtype=np.uint32)

            for output_class, target_class in zip(output_classes, target_classes):
                # calculate metrics for each frame
                # calculate using confusion matrix or dirctly using definition
                cm = calculate_confusion_matrix_from_arrays(output_class, target_class, num_classes)
                iou_mRecords.update_record(calculate_iou(cm))
                dice_mRecords.update_record(calculate_dice(cm))
                cm_b += cm

                ######## calculate directly using definition ##########
                # iou_mRecords.update_record(iou_multi_np(target_class, output_class))
                # dice_mRecords.update_record(dice_multi_np(target_class, output_class))

            # accumulate batch metrics to engine state
            engine.state.epoch_metrics['confusion_matrix'] += cm_b
            engine.state.epoch_metrics['iou'].merge(iou_mRecords)
            engine.state.epoch_metrics['dice'].merge(dice_mRecords)

            return_dict = {
                'loss': loss.item(),
                'output': outputs,
                'output_argmax': output_argmaxs,
                'target': targets,
                # for monitoring
                'iou': iou_mRecords,
                'dice': dice_mRecords,
            }

            return return_dict

    # validator engine
    validator = engine.Engine(valid_step)

    # monitor loss
    valid_ra_loss = imetrics.RunningAverage(output_transform=lambda x: x['loss'], alpha=0.98)
    valid_ra_loss.attach(validator, 'valid_ra_loss')

    # monitor validation loss over epoch
    valid_loss = imetrics.Loss(loss_func, output_transform=lambda x: (x['output'], x['target']))
    valid_loss.attach(validator, 'valid_loss')

    # monitor <data> mean metrics
    valid_data_miou = imetrics.RunningAverage(output_transform=lambda x: x['iou'].data_mean()['mean'], alpha=0.98)
    valid_data_miou.attach(validator, 'mIoU')
    valid_data_mdice = imetrics.RunningAverage(output_transform=lambda x: x['dice'].data_mean()['mean'], alpha=0.98)
    valid_data_mdice.attach(validator, 'mDice')

    # show metrics on progress bar (after every iteration)
    valid_pbar = c_handlers.ProgressBar(persist=True, dynamic_ncols=True)
    valid_metric_names = ['valid_ra_loss', 'mIoU', 'mDice']
    valid_pbar.attach(validator, metric_names=valid_metric_names)


    @validator.on(engine.Events.STARTED)
    def validator_start_callback(engine):
        pass

    @validator.on(engine.Events.EPOCH_STARTED)
    def validator_epoch_start_callback(engine):
        engine.state.epoch_metrics = {
            # directly use definition to calculate
            'iou': MetricRecord(),
            'dice': MetricRecord(),
            'confusion_matrix': np.zeros((num_classes, num_classes), dtype=np.uint32),
        }

    # evaluate after iter finish
    @validator.on(engine.Events.ITERATION_COMPLETED)
    def validator_iter_comp_callback(engine):
        pass

    # evaluate after epoch finish
    @validator.on(engine.Events.EPOCH_COMPLETED)
    def validator_epoch_comp_callback(engine):

        # log monitored epoch metrics
        epoch_metrics = engine.state.epoch_metrics

        ######### NOTICE: Two metrics are available but different ##########
        ### 1. mean metrics for all data calculated by confusion matrix ####

        '''
        compared with using confusion_matrix[1:, 1:] in original code,
        we use the full confusion matrix and only present non-background result
        '''
        confusion_matrix = epoch_metrics['confusion_matrix']  # [1:, 1:]
        ious = calculate_iou(confusion_matrix)
        dices = calculate_dice(confusion_matrix)

        mean_ious = np.mean(list(ious.values()))
        mean_dices = np.mean(list(dices.values()))
        std_ious = np.std(list(ious.values()))
        std_dices = np.std(list(dices.values()))

        logging_logger.info('mean IoU: %.3f, std: %.3f, for each class: %s' %
                            (mean_ious, std_ious, ious))
        logging_logger.info('mean Dice: %.3f, std: %.3f, for each class: %s' %
                            (mean_dices, std_dices, dices))

        ### 2. mean metrics for all data calculated by definision ###
        iou_data_mean = epoch_metrics['iou'].data_mean()
        dice_data_mean = epoch_metrics['dice'].data_mean()

        logging_logger.info('data (%d) mean IoU: %.3f, std: %.3f' %
                            (len(iou_data_mean['items']), iou_data_mean['mean'], iou_data_mean['std']))
        logging_logger.info('data (%d) mean Dice: %.3f, std: %.3f' %
                            (len(dice_data_mean['items']), dice_data_mean['mean'], dice_data_mean['std']))


    # log interal variables(attention maps, outputs, etc.) on validation
    def tb_log_valid_iter_vars(engine, logger, event_name):
        log_tag = 'valid_iter'
        output = engine.state.output
        batch_size = output['output'].shape[0]
        res_grid = tvutils.make_grid(torch.cat([
            output['output_argmax'].unsqueeze(1),
            output['target'].unsqueeze(1),
        ]), padding=2,
            normalize=False,  # show origin image
            nrow=batch_size).cpu()

        logger.writer.add_image(tag='%s (outputs, targets)' % (log_tag), img_tensor=res_grid)


    def tb_log_valid_epoch_vars(engine, logger, event_name):
        log_tag = 'valid_iter'
        # log monitored epoch metrics
        epoch_metrics = engine.state.epoch_metrics
        confusion_matrix = epoch_metrics['confusion_matrix']  # [1:, 1:]
        ious = calculate_iou(confusion_matrix)
        dices = calculate_dice(confusion_matrix)

        mean_ious = np.mean(list(ious.values()))
        mean_dices = np.mean(list(dices.values()))
        logger.writer.add_scalar('mIoU', mean_ious, engine.state.epoch)
        logger.writer.add_scalar('mIoU', mean_dices, engine.state.epoch)

    if args.tb_log:
        # log internal values
        tb_logger.attach(validator, log_handler=tb_log_valid_iter_vars,
                         event_name=engine.Events.ITERATION_COMPLETED)
        tb_logger.attach(validator, log_handler=tb_log_valid_epoch_vars,
                         event_name=engine.Events.EPOCH_COMPLETED)
        # tb_logger.attach(validator, log_handler=OutputHandler('valid_iter', valid_metric_names),
        #     event_name=engine.Events.ITERATION_COMPLETED)
        tb_logger.attach(validator, log_handler=OutputHandler('valid_epoch', ['valid_loss']),
                         event_name=engine.Events.EPOCH_COMPLETED)

    # score function for model saving
    ckpt_score_function = lambda engine: \
        np.mean(list(calculate_iou(engine.state.epoch_metrics['confusion_matrix']).values()))
    # ckpt_score_function = lambda engine: engine.state.epoch_metrics['iou'].data_mean()['mean']

    ckpt_filename_prefix = 'model'

    # model saving handler
    model_ckpt_handler = handlers.ModelCheckpoint(
        dirname=args.model_save_dir,
        filename_prefix=ckpt_filename_prefix,
        score_function=ckpt_score_function,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True,
        atomic=True)

    validator.add_event_handler(event_name=engine.Events.EPOCH_COMPLETED,
                                handler=model_ckpt_handler,
                                to_save={
                                    'model': model,
                                })

    # early stop
    # trainer=trainer, but should be handled by validator
    early_stopping = handlers.EarlyStopping(patience=args.es_patience,
                                            score_function=ckpt_score_function,
                                            trainer=trainer
                                            )

    validator.add_event_handler(event_name=engine.Events.EPOCH_COMPLETED,
                                handler=early_stopping)

    # evaluate after epoch finish
    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def trainer_epoch_comp_callback(engine):
        validator.run(valid_loader)

    trainer.run(train_loader, max_epochs=args.max_epochs)

    if args.tb_log:
        # close tb_logger
        tb_logger.close()

    return trainer.state.metrics_records





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
            Normalize(p=1),
            # optional transforms
            # Resize(height=args.input_height, width=args.input_width, p=1),
            # CenterCrop(height=args.input_height, width=args.input_width, p=1)
            RandomCrop(height=args.input_height, width=args.input_width, p=1),
        ]

    valid_transform_ops = [
        Normalize(p=1),
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
    parser.add_argument('--model', type=str, default='UNet',
                        help='model for segmentation.')

    parser.add_argument('--mf', type=bool, default=True,
                        help='whether to enable Motion Flow for attention map generation. This option is only valid for TAPNet based models. Enable this option to use MF-TAPNet')

    parser.add_argument('--max_epochs', type=int, default=20,
                        help='max epochs for training.')

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate.')

    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='learning rate decay.')

    parser.add_argument('--lr_decay_epochs', type=int, default=5,
                        help='number of epochs for every learning rate decay.')

    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay.')

    parser.add_argument('--es_patience', type=int, default=20,
                        help='early stop patience')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size.')

    parser.add_argument('--labeled_bs', type=int, default=3, help='labeled_batch_size per gpu')

    parser.add_argument('--jaccard_weight', type=float, default=0.0,
                        help='jaccard weight [0.0, 1.0] in loss function.')

    parser.add_argument('--input_height', type=int, default=256,
                        help='input image height.')

    parser.add_argument('--input_width', type=int, default=320,
                        help='input image width.')


    # dirs
    parser.add_argument('--root', type=str, default='./data',
                        help='train data directory.')

    parser.add_argument('--model_save_dir', type=str, default='./model',
                        help='model save dir.')

    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='model checkpoint directory.')

    # pytorch settings
    parser.add_argument('--device_ids', type=int, default=[0, 1, 2, 3], nargs='+',
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
    parser.add_argument('--max_iterations', type=int, default=600, help='maximum epoch number to train')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    args = parser.parse_args()
    main(args)
