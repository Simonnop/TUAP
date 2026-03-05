import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

from .base_attack import BaseAttack

class BaseGlobalAttack(BaseAttack):
    """
    Base class for all attacks.
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mae', epoch=10, decay=1, time_window=None, drop_ratio=0., alpha_times=1):
        # 父类初始化
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, alpha_times)
        # 补充
        self.time_window = time_window
        self.drop_ratio = drop_ratio

    def forward(self, origin_x, x, y=None, seq_x_mark=None, seq_y_mark=None, start = None,**kwargs):
        """
        The general attack procedure
        """

        # 保持输入在 CPU；仅在需要计算的分块搬到 GPU，避免一次性将巨大张量放入显存
        origin_x = origin_x.float()
        x = x.float()
        if y is not None:
            y = y.float()
        if seq_x_mark is not None:
            seq_x_mark = seq_x_mark.float()
        if seq_y_mark is not None:
            seq_y_mark = seq_y_mark.float()

        # Initialize adversarial perturbation
        # 给 data 升一维度
        if len(origin_x.shape) == 2:
            origin_x = origin_x.unsqueeze(0)  # 从 (len, var_num) 升维到 (1, len, var_num)

        # 给定初始扰动
        if start is not None:
            delta = start   
        else:
            delta = self.init_delta(origin_x)

        delta = delta.detach()
        data_len = origin_x.size(1)
        sample_num = data_len - self.time_window + 1
        momentum = [0 for _ in range(data_len)]

        # 攻击轮数
        for epoch in range(self.epoch):
            # 
            print(f"第 {epoch + 1}/{self.epoch} 轮攻击")
            # 单独算每个点的梯度, 算 delta
            for i in tqdm(range(data_len), desc="攻击迭代进度"):

                # 生成样本遮罩
                indicator = torch.zeros_like(origin_x)
                # (1, len, var_num)
                # 将遍历到的时间位置进行标记
                indicator[:, i, :] = 1
                sample_indicator = self.original_data_to_sample(indicator, sample_num, self.time_window)

                # 寻找相关样本,即 indicator 存在 1 的样本
                relevant_indices = []
                for j in range(sample_num):
                    if torch.sum(sample_indicator[j]) > 0:
                        relevant_indices.append(j)
                # print(f"相关样本: {relevant_indices}")

                # 如果有 drop_ratio 则随机丢弃相关样本
                keep_num = int(len(relevant_indices) * (1 - self.drop_ratio))
                keep_num = 1 if keep_num == 0 else keep_num
                random_indices = torch.randperm(len(relevant_indices))
                relevant_indices = [relevant_indices[i] for i in random_indices[:keep_num]]
                # print(f"相关样本: {relevant_indices}")

                # 创建新的delta_i张量而不是使用切片
                # (1, 1, var_num)
                delta_i = delta[0, i, :].clone().detach().unsqueeze(0).unsqueeze(0)
                delta_i.requires_grad = True
                # 扩展 delta_i, 方便矩阵相乘取到对应的 delta
                delta_expand = delta_i.repeat(len(relevant_indices), 1, 1)  # (len(relevant_indices), 1, var_num)

                # 采用分块计算，避免一次性前向占满显存
                chunk_size = getattr(self.args, 'global_chunk_size', None)
                if chunk_size is None or chunk_size <= 0:
                    # 退化为使用 batch_size 作为分块大小；若未设置则默认 64
                    chunk_size = getattr(self.args, 'batch_size', 64) or 64

                grad_sum = torch.zeros_like(delta_i)
                for k in range(0, len(relevant_indices), chunk_size):
                    idx_chunk = relevant_indices[k:k+chunk_size]
                    # 将当前分块搬到 GPU 进行计算
                    x_chunk = x[idx_chunk].to(self.device)
                    si_chunk = sample_indicator[idx_chunk].to(self.device)
                    delta_expand_chunk = delta_i.repeat(len(idx_chunk), 1, 1)

                    perturbated_data_chunk = x_chunk + si_chunk * delta_expand_chunk
                    prediction, true = self.get_prediction(
                        self.transform(data=perturbated_data_chunk, momentum=momentum),
                        y[idx_chunk],
                        seq_x_mark[idx_chunk],
                        seq_y_mark[idx_chunk]
                    )
                    object_value_chunk = self.get_object_value(prediction.reshape(-1,1), true.reshape(-1,1))
                    grad_chunk = self.get_grad(object_value_chunk, delta_i)
                    grad_sum = grad_sum + grad_chunk
                    # 及时释放分块引用
                    del x_chunk, si_chunk, delta_expand_chunk, perturbated_data_chunk, prediction, true, object_value_chunk, grad_chunk

                    # GC
                    torch.cuda.empty_cache()

                grad = grad_sum

                # 算动量
                momentum[i] = self.get_momentum(grad, momentum[i])
                # 算 delta
                delta_i = self.update_delta(delta_i, momentum[i], self.alpha)
                delta_i.detach()
                delta[0,i,:] = delta_i[0,0,:]

        return delta.detach()[0,:,:]

    def original_data_to_sample(self, original_data, sample_num, time_window):
        # 将原始数据转换为样本
        # sample: (sample_num, time_window, feature_num)
        sample = torch.zeros((sample_num, time_window, original_data.shape[2])).to(original_data.device)
        for i in range(sample_num):
            sample[i] = original_data[0,i:i+time_window,:]
        return sample


class GlobalGradientAccumulationAttack(BaseGlobalAttack):
    """
    Global Gradient Accumulation Attack (GGAA) implementation.
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
        random_start=False
    ):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, time_window, drop_ratio, alpha_times)
        self.random_start = random_start

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

        if self.random_start:
            delta = torch.zeros_like(origin_x, device=self.device).uniform_(-self.epsilon, self.epsilon).requires_grad_(True)
        else:
            delta = torch.zeros_like(origin_x, device=self.device, requires_grad=True)
        momentum = torch.zeros_like(delta)

        for epoch in range(self.epoch):
            global_grad = torch.zeros_like(delta)
            print(f"第 {epoch + 1}/{self.epoch} 轮攻击")
            # 若有 drop_ratio 则随机丢弃部分样本
            if self.drop_ratio > 0:
                keep_num = int(sample_num * (1 - self.drop_ratio))
                keep_num = 1 if keep_num == 0 else keep_num
                sample_indices = torch.randperm(sample_num)[:keep_num].tolist()
            else:
                sample_indices = list(range(sample_num))
            # 单独算每个点的梯度, 算 delta
            for idx in tqdm(sample_indices, desc="攻击迭代进度"):
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
                global_grad[:, start:end, :] += grad.detach()

                del perturbed_window, y_slice, x_mark_slice, y_mark_slice, prediction, true, loss, grad

            norm = global_grad.abs().sum()
            g_norm = global_grad / (norm + 1e-12)
            momentum = self.decay * momentum + g_norm

            update = delta + self.alpha * momentum.sign()
            delta = torch.clamp(update, -self.epsilon, self.epsilon).detach().requires_grad_(True)
            momentum = momentum.detach()

        return delta.detach()[0]