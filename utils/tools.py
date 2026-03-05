import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import logging

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    """三种方式调整学习率

    Args:
        optimizer (torch.optim): 优化器对象
        epoch (int): 当前epoch
        args (): 需要lradj、learning_rate、train_epochs
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    """早停类 回调时会判断性地储存检查点
    
        每获得一次验证损失 回调一次
    """
    def __init__(self, patience=7, verbose=False, delta=0, save_dict_only=False, logger=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_dict_only = save_dict_only
        self.logger = logger

    def printLogs(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, self.save_dict_only)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.printLogs(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, self.save_dict_only)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, save_dict_only):
        if self.verbose:
            self.printLogs(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if save_dict_only:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        else:
            torch.save(model, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """结果可视化

    Args:
        true (_type_): 真实值
        preds (_type_, optional): 预测值. Defaults to None.
        name (str, optional): 默认保存路径及格式. Defaults to './pic/test.pdf'.
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def setup_logger(name, log_file):
   # 创建或获取一个名为 'my_logger' 的 logger
    logger = logging.getLogger(name)

    # 设置最低的日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.DEBUG)

    # 创建一个文件处理程序，写入日志到 'app.log' 文件中
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # 也可以为每个处理程序单独设置日志级别

    # 创建一个控制台处理程序，打印日志到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上级别的日志

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s -  %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理程序添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 移除默认的处理程序（如果有的话），避免重复输出
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
    

class PrintLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

