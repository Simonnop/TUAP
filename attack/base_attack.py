import torch
import torch.nn as nn

import numpy as np

class BaseAttack():
    """
    Base class for all attacks.
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mse', epoch=10, decay=1, alpha_times=1):
        """
        Initialize the hyperparameters

        Arguments:
            attack (str): the name of attack.
            model_name (str): the name of surrogate model for attack.
            epsilon (float): the perturbation budget.
            norm (str): the norm of perturbation, l2/linfty.
            device (torch.device): the device for data.
            args (dict): the arguments for attack.
            metrics (str): the metrics for attack, default is 'mse'.
            epoch (int): the number of iterations.
            decay (float): the decay factor for momentum calculation.
            alpha_times (float): the multiplier for alpha calculation, alpha = epsilon / epoch * alpha_times
        """

        if norm not in ['l2', 'linfty']:
            raise Exception("Unsupported norm {}".format(norm))
        self.norm = norm
        self.model = model
        self.args = args
        if metrics not in ['mse', 'mae', 'rmse', 'mape', 'mspe']:
            raise Exception("Unsupported metrics {}".format(metrics))
        self.metrics = metrics
        self.epsilon = epsilon
        self.device = next(self.model.parameters()).device if device is None else device

        # 根据攻击类型, 初始化攻击参数
        self.attack = attack
        if self.attack == 'FGSM' or self.attack == 'AAIM' or self.attack == 'ADJM':
            self.random_start = False
            self.alpha = self.epsilon * alpha_times
            self.epoch = 1
            self.decay = 0
        elif self.attack == 'ATSG':
            self.random_start = False
            self.alpha = self.epsilon * alpha_times
            self.epoch = 1
            self.decay = 0
        elif self.attack == 'BIM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = 0
        elif self.attack == 'MIFGSM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = decay
        elif self.attack == 'PGD':
            self.random_start = True
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = 0
        elif self.attack == 'GGAA':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = decay
        elif self.attack in ('GTW', 'GlobalTimestampWise'):
            self.random_start = False
            self.alpha = self.epsilon
            self.epoch = 1
            self.decay = 0
        elif self.attack in ('GTW_Fix'):
            self.random_start = True
            self.alpha = self.epsilon
            self.epoch = 1
            self.decay = 0
        elif self.attack in ('EMPTY_GLOBAL',):
            # 用于 global 流程的固定开销基线；此处仅为合法化该 attack 名称
            self.random_start = False
            self.alpha = self.epsilon
            self.epoch = 1
            self.decay = 0
        elif self.attack == 'TCA':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = 0
        elif self.attack == 'NIFGSM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = decay
        elif self.attack == 'VMIFGSM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = decay
        elif self.attack == 'GGAA_NIFGSM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = decay
        elif self.attack == 'GGAA_VMIFGSM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = decay
        elif self.attack == 'GGAA_FGSM':
            self.random_start = False
            self.alpha = self.epsilon * alpha_times
            self.epoch = 1
            self.decay = 0
        elif self.attack == 'GGAA_BIM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = 0
        elif self.attack == 'GGAA_PGD':
            self.random_start = True
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = 0
        elif self.attack == 'VNIFGSM' or self.attack == 'GGAA_VNIFGSM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = decay
        elif self.attack == 'IEFGSM' or self.attack == 'GGAA_IEFGSM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = decay
        elif self.attack == 'AIFGTM' or self.attack == 'GGAA_AIFGTM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times # Note: AIFGTM calculates alpha dynamically, but base init is fine
            self.epoch = epoch
            self.decay = decay
        elif self.attack == 'AdaMSI_FGM' or self.attack == 'GGAA_AdaMSI_FGM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = decay
        elif self.attack == 'GIFGSM' or self.attack == 'GGAA_GIFGSM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = decay
        elif self.attack == 'PIFGSM' or self.attack == 'GGAA_PIFGSM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = decay
        # elif self.attack == 'CW':
        #     self.random_start = False
        #     self.alpha = alpha_times # 对于 CW，alpha 直接作为学习率使用
        #     self.epoch = epoch
        #     self.decay = 0
        elif self.attack == 'BO':
            pass
        elif self.attack in ('GGAA_First', 'GGAA_Last', 'GGAA_Random'):
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = decay
        else:
            raise ValueError("Invalid attack: {}".format(self.attack))
        

    def forward(self, data, y=None, seq_x_mark=None, seq_y_mark=None, **kwargs):
        """
        The general attack procedure
        
        单变量预测

        """

        # 使用 cuDNN RNN 进行反向传播需要模型处于训练模式；参数已冻结，不会产生参数梯度
        self.model.train()

        # Initialize adversarial perturbation
        # delta = self.init_delta(data)
        # 给 data 升一维度
        if len(data.shape) == 2:
            data = data.unsqueeze(0)  # 从 (T, F) 升维到 (1, T, F)
        # 确保 data 在同一设备上
        data = data.float().to(self.device)
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            
            # attack_data (N, K, T): 可以攻击的辅助变量
            # unattack_data (N, 1, T): 不可攻击的变量
            perturbated_data = delta + data
            
            # 直接拿预测值
            prediction, true = self.get_prediction(self.transform(data=perturbated_data, momentum=momentum), y, seq_x_mark, seq_y_mark)

            # 算目标函数值, 根据攻击存在正负之分
            object_value = self.get_object_value(prediction, true)

            # 算梯度
            grad = self.get_grad(object_value, delta)

            # 算动量
            momentum = self.get_momentum(grad, momentum)

            # 算 delta
            delta = self.update_delta(delta, momentum, self.alpha)

        # 返回前断开计算图；如调用方需要常驻保存，建议转回 CPU
        delta, true = delta.detach(), true.detach()
        return delta, true

    def get_prediction(self, x, y, seq_x_mark, seq_y_mark, **kwargs):

        batch_x = x.float().to(self.device)
        batch_y = y.float().to(self.device)

        batch_x_mark = seq_x_mark.float().to(self.device)
        batch_y_mark = seq_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, :].to(self.device)
        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

        outputs = outputs[:, :, f_dim:]
        batch_y = batch_y[:, :, f_dim:]

        return outputs, batch_y

    def get_object_value(self, prediction: torch.Tensor, y = None):
        return self.calcu_loss(prediction, y)[self.metrics]

    def get_grad(self, object_value, delta, **kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(object_value, inputs=delta, retain_graph=False, create_graph=False)[0]

    def get_momentum(self, grad, momentum, **kwargs):
        """
        The momentum calculation
        """
        # return momentum * self.decay + grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))
        return momentum * self.decay + grad / (grad.abs().mean(dim=(1,2), keepdim=True))

    def init_delta(self, data, **kwargs):
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            if self.norm == "linfty":
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                # to time series
                delta = torch.zeros_like(data)
                delta.normal_(-self.epsilon, self.epsilon)
                # 应用 mask 到 delta 上, 不要算进二范数里
                d_flat = delta.view(delta.size(0), -1)
                # n = d_flat.norm(p=2, dim=-1).view(delta.size(0), 1, 1, 1)
                n = d_flat.norm(p=2, dim=-1).view(delta.size(0), 1, 1)
                r = torch.zeros_like(data).uniform_(0, 1)
                delta *= r / (n + 1e-20) * self.epsilon 

            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        delta.requires_grad = True
        return delta

    def update_delta(
        self, delta: torch.Tensor, grad, alpha: torch.Tensor, **kwargs
    ):
        if self.norm == "linfty":
            delta = torch.clamp(
                delta + alpha * grad.sign(), -self.epsilon, self.epsilon
            )
        else:
            # grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (
                (delta + scaled_grad * alpha)
                .view(delta.size(0), -1)
                .renorm(p=2, dim=0, maxnorm=self.epsilon)
                .view_as(delta)
            )
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        return delta.detach().requires_grad_(True)

    def transform(self, data, **kwargs):
        return data

    def __call__(self, *input, **kwargs):
        # self.model.eval()
        return self.forward(*input, **kwargs)

    def calcu_loss(self, pred, true):

        # 确保 pred 和 true 形状相同
        assert pred.shape == true.shape, f"预测值形状 {pred.shape} 与真实形状 {true.shape} 不匹配"

        # 计算各种损失指标
        diff = pred - true
        abs_diff = torch.abs(diff)
        abs_true = torch.abs(true)
            
        # MSE
        mse = torch.mean(diff * diff)
        # MAE
        mae = torch.mean(abs_diff)
        # RMSE
        rmse = torch.sqrt(mse)
        # MAPE
        mape = torch.mean(abs_diff / (abs_true + 1e-8))
        # MSPE
        mspe = torch.mean((diff / (true + 1e-8)) ** 2)
            
        return {
            'mse': mse,
            'mae': mae, 
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe
        }