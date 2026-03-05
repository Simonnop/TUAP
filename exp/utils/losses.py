# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch
import torch.nn as nn
import numpy as np
import joblib

def calcu_loss(pred, true, data_name:str=None):
        if data_name is not None:
            # 加载 scaler
            # /home/suruixian/experiments/TS_Attack/data/scaler/spain_y_scaler.joblib
            scaler = joblib.load(f'./data/scaler/{data_name.lower()}_y_scaler.joblib')
            # 将 pred 和 true 转换为 numpy 数组
            pred_numpy = pred.detach().cpu().numpy().reshape(-1, 1)
            true_numpy = true.detach().cpu().numpy().reshape(-1, 1)
            
            # 使用 scaler 进行反归一化
            pred_numpy = scaler.inverse_transform(pred_numpy)
            true_numpy = scaler.inverse_transform(true_numpy)
        else:
            pred_numpy = pred.detach().cpu().numpy().reshape(-1, 1)
            true_numpy = true.detach().cpu().numpy().reshape(-1, 1)
        
        # 确保 pred 和 true 形状相同
        assert pred.shape == true.shape, f"预测值形状 {pred.shape} 与真实值形状 {true.shape} 不匹配"

        pred_tensor = torch.FloatTensor(pred_numpy).squeeze()
        true_tensor = torch.FloatTensor(true_numpy).squeeze()

        # 计算各种损失指标
        diff = pred_tensor - true_tensor
        abs_diff = torch.abs(diff)
        abs_true = torch.abs(true_tensor)
        
        # MSE
        mse = torch.mean(diff * diff)
        # MAE
        mae = torch.mean(abs_diff)
        # RMSE
        rmse = torch.sqrt(mse)
        # MAPE
        mape = torch.mean(abs_diff / (abs_true + 1e-8))
        # MSPE
        mspe = torch.mean((diff / (true_tensor + 1e-8)) ** 2)
        
        return {
            'mse': mse.item(),
            'mae': mae.item(), 
            'rmse': rmse.item(),
            'mape': mape.item(),
            'mspe': mspe.item()
        }


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.Tensor:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.Tensor:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.Tensor:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)
