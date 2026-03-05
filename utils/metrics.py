import numpy as np
from tqdm import tqdm
import torch


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / (true + 1e-8)))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / (true + 1e-8)))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


class reshapeBeforeMetric():
    """有时模型输出与真实值不同形状，这里进行转换

    Args:
        pred (np.array): 预测值
        true (np.array): 真实值

    Returns:
        np.array, np.array: 预测值，真实值
    """
    def __init__(self, logger=None):
        self.modified = False  # 让后文输出只输出一次 避免每次调用都输出
        self.logger = logger

    def printLogs(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            tqdm.write(msg)

    def __call__(self, args, pred, true):
        if pred.shape[2] != true.shape[2]:
            B, T, C = true.shape
            match args:
                case args if args.graph_flag is not None:
                    if not self.modified:
                        self.printLogs(f'预测结果({pred.shape[2]})与真实值({true.shape[2]})形状不匹配！目前匹配图结构。')
                    pred = pred.reshape(B, T, int(args.graph_flag), -1)
                    pred = pred[:, :, :, -1]
                    pred = pred.reshape(pred.shape[0], pred.shape[1], -1)
                    self.modified = True
                case _:
                    if not self.modified:
                        self.printLogs(f'预测结果({pred.shape[2]})与真实值({true.shape[2]})形状不匹配！目前取预测结果末尾值。')
                    pred = pred[:, :, -C:]
                    self.modified = True
        else:
            if not self.modified:
                self.printLogs('预测结果与真实值形状匹配。')
            self.modified = True
        return pred, true


def adapted_metric(args, pred, true):
    """输入预测和真实值，返回mae、mse、rmse、mape、mspe。相比metric可更加方便地计算和更改评价指标

    Args:
        pred (np.array): 预测值
        true (np.array): 真实值

    Returns:
        double: metric_names, (loss, mae, mse, rmse, mape, mspe)
    """
    metric_names = ['loss', 'MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
    loss = args.criterion(torch.tensor(pred), torch.tensor(true)).numpy()
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return metric_names, (loss, mae, mse, rmse, mape, mspe)
