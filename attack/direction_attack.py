import torch
import torch.nn as nn

import numpy as np

class DirectionAttack():
    """
    direction attack
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mae', epoch=10, decay=1, alpha_times=1):
        """
        Initialize the hyperparameters

        Arguments:
            attack (str): the name of attack.
            model_name (str): the name of surrogate model for attack.
            epsilon (float): the perturbation budget.
            norm (str): the norm of perturbation, l2/linfty.
            device (torch.device): the device for data.
            args (dict): the arguments for attack.
            metrics (str): the metrics for attack, default is 'mae'.
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
        elif self.attack == 'MIFGSM':
            self.random_start = False
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = 0
        elif self.attack == 'PGD':
            self.random_start = True
            self.alpha = self.epsilon / epoch * alpha_times
            self.epoch = epoch
            self.decay = decay
        else:
            raise ValueError("Invalid attack: {}".format(self.attack))
        

    def forward(self, data, y=None, seq_x_mark=None, seq_y_mark=None, direction=None, **kwargs):
        """
        The general attack procedure
        
        单变量预测

        """

        # Initialize adversarial perturbation
        # delta = self.init_delta(data)
        # 给 data 升一维度
        # 对所有输入数据进行升维处理
        if len(data.shape) == 2:
            data = data.unsqueeze(0)  # 从 (T, F) 升维到 (1, T, F)
        if y is not None and len(y.shape) == 2:
            y = y.unsqueeze(0)  # 从 (T, F) 升维到 (1, T, F)
        if seq_x_mark is not None and len(seq_x_mark.shape) == 2:
            seq_x_mark = seq_x_mark.unsqueeze(0)  # 从 (T, D) 升维到 (1, T, D)
        if seq_y_mark is not None and len(seq_y_mark.shape) == 2:
            seq_y_mark = seq_y_mark.unsqueeze(0)  # 从 (T, D) 升维到 (1, T, D)
        
        # 将所有输入数据移到正确的设备上
        data = data.to(self.device)
        if y is not None:
            y = y.to(self.device)
        if seq_x_mark is not None:
            seq_x_mark = seq_x_mark.to(self.device)
        if seq_y_mark is not None:
            seq_y_mark = seq_y_mark.to(self.device)
        
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            
            # attack_data (N, K, T): 可以攻击的辅助变量
            # unattack_data (N, 1, T): 不可攻击的变量
            perturbated_data = delta + data
            
            # 直接拿预测值
            prediction, true = self.get_prediction(self.transform(data=perturbated_data, momentum=momentum), y, seq_x_mark, seq_y_mark)

            # 算目标函数值, 根据攻击存在正负之分
            object_value = self.get_object_value(prediction, direction)

            # 算梯度
            grad = self.get_grad(object_value, delta)

            # 算动量
            momentum = self.get_momentum(grad, momentum)

            # 算 delta
            delta = self.update_delta(delta, momentum, self.alpha)

        delta, true = delta.detach(), true.detach()

        torch.cuda.empty_cache()

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

    def get_object_value(self, prediction: torch.Tensor, direction):
        if direction == 'increase' or direction == 1:
            return prediction.sum()
        elif direction == 'decrease' or direction == 0:
            return -prediction.sum()

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