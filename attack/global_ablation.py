from operator import length_hint
import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

from .global_attack import BaseGlobalAttack

class Ablation(BaseGlobalAttack):
    """
    Ablation.
    """

    def __init__(
        self,
        attack,
        model,
        epsilon,
        norm,
        device=None,
        args=None,
        metrics='mae',
        epoch=10,
        decay=1,
        time_window=None,
        drop_ratio=0.,
        alpha_times=1,
        ablation_type=None
    ):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, time_window, drop_ratio, alpha_times)
        if ablation_type not in ['first', 'last', 'random']:
            raise ValueError(f"Unsupported ablation type: {ablation_type}")
        self.ablation_type = ablation_type

    def forward(self, origin_x, x, y=None, seq_x_mark=None, seq_y_mark=None, **kwargs):
        if y is None or seq_x_mark is None or seq_y_mark is None:
            raise ValueError("GGAA攻击需要 y、seq_x_mark 和 seq_y_mark")

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

        space = window_len + 1
        feature_num = x.size(2)
        length = origin_x.size(1)

        delta = torch.zeros_like(origin_x, device=self.device, requires_grad=True)
        momentum = torch.zeros_like(delta)

        for epoch in range(self.epoch):
            # 梯度累积池
            global_grad = torch.zeros_like(delta)
            gradient_pool = torch.zeros_like(x, device=self.device)
            print(f"第 {epoch + 1}/{self.epoch} 轮攻击")
            # 单独算每个点的梯度, 算 delta
            for idx in tqdm(range(sample_num), desc="攻击迭代进度"):
                start = idx
                end = start + window_len
                delta_slice = delta[:, start:end, :]
                perturbed_window = origin_x[:, start:end, :] + delta_slice

                y_slice = y[idx:idx + 1].to(self.device)
                x_mark_slice = seq_x_mark[idx:idx + 1].to(self.device)
                y_mark_slice = seq_y_mark[idx:idx + 1].to(self.device)

                prediction, true = self.get_prediction(perturbed_window, y_slice, x_mark_slice, y_mark_slice)
                loss = self.get_object_value(prediction, true)
                grad = torch.autograd.grad(loss, delta_slice, retain_graph=False, create_graph=False)[0]

                gradient_pool[idx, :, :] = grad.detach()

                del perturbed_window, y_slice, x_mark_slice, y_mark_slice, prediction, true, loss, grad

            candidate_space = self.sample_to_candidate(gradient_pool, sample_num, length, space, feature_num, window_len)

            if self.ablation_type == 'first':
                global_grad[:, :, :] = candidate_space[:, 0, :]
            elif self.ablation_type == 'last':
                print(f"candidate_space[:, -1, :]: {candidate_space[:, -2, :]}")
                global_grad[:, :, :] = candidate_space[:, -2, :]
            elif self.ablation_type == 'random':
                n_candidates = candidate_space.size(1)
                for i in range(candidate_space.size(0)):
                    random_index = torch.randint(0, n_candidates, (1,), device=candidate_space.device).item()
                    global_grad[:, i, :] = candidate_space[i, random_index, :]

            norm = global_grad.abs().sum()
            g_norm = global_grad / (norm + 1e-12)
            momentum = self.decay * momentum + g_norm

            update = delta + self.alpha * momentum.sign()
            delta = torch.clamp(update, -self.epsilon, self.epsilon).detach().requires_grad_(True)
            momentum = momentum.detach()

        return delta.detach()[0]

    def sample_to_candidate(self, sample, sample_num, len, space, feature_num, time_window):
        '''
        将滑动窗口切割的样本 -> 梯形决策空间
        '''
        # 一次性转到 CPU，避免循环内 26 万次 GPU->CPU 拷贝
        sample = sample.cpu()
        origin_data_with_candidate = torch.zeros((len, space, feature_num))
        
        # 矩阵 -> 梯形决策空间
        dot_num_index = torch.zeros(len)
        for i in tqdm(range(sample_num), desc="形成决策空间", dynamic_ncols=True):
            for j in range(time_window):
                num_index = int(dot_num_index[i + j].item())
                origin_data_with_candidate[i + j, num_index, :] = sample[i, j, :]
                dot_num_index[i + j] = dot_num_index[i + j] + 1

        # 填补梯形前后的角
        for i in range(time_window):
            space = i + 1
            for j in range(time_window):
                # 从space随机抽一个数
                random_idx = torch.randint(0, space, (1,)).item()
                # 前面的角
                origin_data_with_candidate[i, j, :] = origin_data_with_candidate[i, random_idx, :]
                # 后面的角
                origin_data_with_candidate[len - i - 1, j, :] = origin_data_with_candidate[len -i - 1, random_idx, :] 
        
        return origin_data_with_candidate.to(self.device)