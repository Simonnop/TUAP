import torch
import torch.nn.functional as F
import numpy as np
from ..base_attack import BaseAttack
from ..global_attack import GlobalGradientAccumulationAttack
from tqdm import tqdm


class PIFGSM(BaseAttack):
    """
    PI-FGSM Attack (Window-wise)
    'Patch-wise Attack for Fooling Deep Neural Network (ECCV 2020)'
    适配时序场景：沿时间维度做 1D patch 投影
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mse', epoch=10, decay=0., alpha_times=1, kern_size=3, gamma=16.0, beta=10.0):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, alpha_times)
        self.kern_size = kern_size
        self.gamma = gamma / 255.0  # 与 CV 版本一致，时序数据 scale 类似
        self.beta = beta

    def project_kern_1d(self, kern_size, num_channels):
        """1D 投影核：沿时间维度，中心为 0，其余均匀"""
        kern = np.ones(kern_size, dtype=np.float32) / (kern_size - 1)
        kern[kern_size // 2] = 0.0
        # (num_channels, 1, kern_size) for conv1d groups=num_channels
        stack_kern = np.tile(kern.reshape(1, 1, -1), (num_channels, 1, 1))
        return torch.tensor(stack_kern, dtype=torch.float32).to(self.device), kern_size // 2

    def project_noise_1d(self, x, stack_kern, padding_size):
        """x: (N, T, F) -> 转 (N, F, T) 做 conv1d"""
        # x: (N, T, F) -> (N, F, T)
        x = x.permute(0, 2, 1)
        x = F.conv1d(x, stack_kern, padding=padding_size, groups=x.size(1))
        return x.permute(0, 2, 1)  # (N, T, F)

    def update_delta_pi(self, delta, grad, alpha, projection):
        """带 patch 投影的 delta 更新"""
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.sign() + projection, -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha + projection).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        return delta.detach().requires_grad_(True)

    def forward(self, data, y=None, seq_x_mark=None, seq_y_mark=None, **kwargs):
        self.model.train()

        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        data = data.float().to(self.device)
        delta = self.init_delta(data)

        num_channels = data.size(-1)
        stack_kern, padding_size = self.project_kern_1d(self.kern_size, num_channels)

        momentum = 0.0
        amplification = 0.0

        for _ in range(self.epoch):
            perturbated_data = delta + data
            prediction, true = self.get_prediction(perturbated_data, y, seq_x_mark, seq_y_mark)
            object_value = self.get_object_value(prediction, true)
            grad = self.get_grad(object_value, delta)

            momentum = self.get_momentum(grad, momentum)

            amplification += self.beta * self.alpha * momentum.sign()
            cut_noise = torch.clamp(torch.abs(amplification) - self.epsilon, 0, 10000.0) * torch.sign(amplification)
            projection = self.gamma * torch.sign(self.project_noise_1d(cut_noise, stack_kern, padding_size))
            amplification = amplification + projection

            delta = self.update_delta_pi(delta, momentum, self.beta * self.alpha, projection)

        delta, true = delta.detach(), true.detach()
        return delta, true


class GGAA_PIFGSM(GlobalGradientAccumulationAttack):
    """
    GGAA with PI-FGSM (Global)
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mae', epoch=10, decay=0., time_window=None, drop_ratio=0., alpha_times=1, kern_size=3, gamma=16.0, beta=10.0):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, time_window, drop_ratio, alpha_times)
        self.kern_size = kern_size
        self.gamma = gamma / 255.0
        self.beta = beta

    def project_kern_1d(self, kern_size, num_channels):
        kern = np.ones(kern_size, dtype=np.float32) / (kern_size - 1)
        kern[kern_size // 2] = 0.0
        stack_kern = np.tile(kern.reshape(1, 1, -1), (num_channels, 1, 1))
        return torch.tensor(stack_kern, dtype=torch.float32).to(self.device), kern_size // 2

    def project_noise_1d(self, x, stack_kern, padding_size):
        x = x.permute(0, 2, 1)
        x = F.conv1d(x, stack_kern, padding=padding_size, groups=x.size(1))
        return x.permute(0, 2, 1)

    def update_delta_pi(self, delta, grad, alpha, projection):
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.sign() + projection, -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha + projection).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        return delta.detach().requires_grad_(True)

    def forward(self, origin_x, x, y=None, seq_x_mark=None, seq_y_mark=None, **kwargs):
        if y is None or seq_x_mark is None or seq_y_mark is None:
            raise ValueError("GGAA_PIFGSM attack requires y, seq_x_mark, and seq_y_mark")

        self.model.train()

        origin_x = origin_x.float()
        if len(origin_x.shape) == 2:
            origin_x = origin_x.unsqueeze(0)
        origin_x = origin_x.to(self.device)

        x = x.float()
        y = y.float()
        seq_x_mark = seq_x_mark.float()
        seq_y_mark = seq_y_mark.float()

        window_len = x.size(1)
        sample_num = x.size(0)
        num_channels = origin_x.size(-1)

        stack_kern, padding_size = self.project_kern_1d(self.kern_size, num_channels)

        delta = torch.zeros_like(origin_x, device=self.device, requires_grad=True)
        momentum = torch.zeros_like(delta)
        amplification = torch.zeros_like(delta)

        for ep in range(self.epoch):
            global_grad = torch.zeros_like(delta)
            print(f"Attack Epoch {ep + 1}/{self.epoch} (GGAA_PIFGSM)")
            for idx in tqdm(range(sample_num), desc="Attack Progress"):
                start, end = idx, idx + window_len
                delta_slice = delta[:, start:end, :]
                perturbed_window = origin_x[:, start:end, :] + delta_slice
                y_slice = y[idx:idx + 1].to(self.device)
                x_mark_slice = seq_x_mark[idx:idx + 1].to(self.device)
                y_mark_slice = seq_y_mark[idx:idx + 1].to(self.device)
                prediction, true = self.get_prediction(perturbed_window, y_slice, x_mark_slice, y_mark_slice)
                loss = self.get_object_value(prediction, true)
                grad = torch.autograd.grad(loss, delta_slice, retain_graph=False, create_graph=False)[0]
                global_grad[:, start:end, :] += grad.detach()
                del perturbed_window, y_slice, x_mark_slice, y_mark_slice, prediction, true, loss, grad

            norm_val = global_grad.abs().sum()
            g_norm = global_grad / (norm_val + 1e-12)
            momentum = self.decay * momentum + g_norm

            amplification = amplification + self.beta * self.alpha * momentum.sign()
            cut_noise = torch.clamp(torch.abs(amplification) - self.epsilon, 0, 10000.0) * torch.sign(amplification)
            projection = self.gamma * torch.sign(self.project_noise_1d(cut_noise, stack_kern, padding_size))
            amplification = amplification + projection

            delta = self.update_delta_pi(delta, momentum, self.beta * self.alpha, projection)
            momentum = momentum.detach()

        return delta.detach()[0]
