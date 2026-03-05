import torch
from ..base_attack import BaseAttack
from ..global_attack import GlobalGradientAccumulationAttack
from tqdm import tqdm


class GIFGSM(BaseAttack):
    """
    GI-FGSM Attack (Window-wise)
    'Boosting the Transferability of Adversarial Attacks with Global Momentum Initialization'
    两阶段：pre_epoch 全局搜索 + main epoch 精调
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mse', epoch=10, decay=1, alpha_times=1, pre_epoch=5, s=10):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, alpha_times)
        self.pre_epoch = pre_epoch
        self.s = s

    def forward(self, data, y=None, seq_x_mark=None, seq_y_mark=None, **kwargs):
        self.model.train()

        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        data = data.float().to(self.device)
        delta = self.init_delta(data)

        momentum = 0

        # 阶段一：pre_epoch 全局搜索，步长 alpha * s
        for _ in range(self.pre_epoch):
            perturbated_data = delta + data
            prediction, true = self.get_prediction(self.transform(data=perturbated_data, momentum=momentum), y, seq_x_mark, seq_y_mark)
            object_value = self.get_object_value(prediction, true)
            grad = self.get_grad(object_value, delta)
            momentum = self.get_momentum(grad, momentum)
            delta = self.update_delta(delta, momentum, self.alpha * self.s)

        # 阶段二：main epoch 精调，步长 alpha
        delta = self.init_delta(data)
        momentum = 0
        for _ in range(self.epoch):
            perturbated_data = delta + data
            prediction, true = self.get_prediction(self.transform(data=perturbated_data, momentum=momentum), y, seq_x_mark, seq_y_mark)
            object_value = self.get_object_value(prediction, true)
            grad = self.get_grad(object_value, delta)
            momentum = self.get_momentum(grad, momentum)
            delta = self.update_delta(delta, momentum, self.alpha)

        delta, true = delta.detach(), true.detach()
        return delta, true


class GGAA_GIFGSM(GlobalGradientAccumulationAttack):
    """
    GGAA with GI-FGSM (Global)
    两阶段：pre_epoch 全局搜索 + main epoch 精调
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mae', epoch=10, decay=1, time_window=None, drop_ratio=0., alpha_times=1, pre_epoch=5, s=10):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, time_window, drop_ratio, alpha_times)
        self.pre_epoch = pre_epoch
        self.s = s

    def forward(self, origin_x, x, y=None, seq_x_mark=None, seq_y_mark=None, **kwargs):
        if y is None or seq_x_mark is None or seq_y_mark is None:
            raise ValueError("GGAA_GIFGSM attack requires y, seq_x_mark, and seq_y_mark")

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

        delta = torch.zeros_like(origin_x, device=self.device, requires_grad=True)
        momentum = torch.zeros_like(delta)

        # 阶段一：pre_epoch 全局搜索
        for ep in range(self.pre_epoch):
            global_grad = torch.zeros_like(delta)
            print(f"Attack Epoch {ep + 1}/{self.pre_epoch} (GGAA_GIFGSM pre_epoch)")
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
            update = delta + (self.alpha * self.s) * momentum.sign()
            delta = torch.clamp(update, -self.epsilon, self.epsilon).detach().requires_grad_(True)
            momentum = momentum.detach()

        # 阶段二：main epoch 精调
        delta = torch.zeros_like(origin_x, device=self.device, requires_grad=True)
        momentum = torch.zeros_like(delta)
        for ep in range(self.epoch):
            global_grad = torch.zeros_like(delta)
            print(f"Attack Epoch {ep + 1}/{self.epoch} (GGAA_GIFGSM)")
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
            update = delta + self.alpha * momentum.sign()
            delta = torch.clamp(update, -self.epsilon, self.epsilon).detach().requires_grad_(True)
            momentum = momentum.detach()

        return delta.detach()[0]
