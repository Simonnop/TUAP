# Global Timestamp-Wise (GTW) 攻击：按时间戳逐点更新扰动的全局攻击

import torch
from tqdm import tqdm

from .global_attack import BaseGlobalAttack


class GlobalTimestampWiseFix(BaseGlobalAttack):
    """
    Global Timestamp-Wise (GTW_Fix) 攻击。
    按时间戳逐点迭代：每个时间点用所有包含该时间戳的滑动窗口样本一次性计算梯度并更新该点的扰动。
    支持分块批处理以提升计算效率。
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
        drop_ratio=0,
        alpha_times=1,
    ):
        super().__init__(
            attack, model, epsilon, norm, device, args,
            metrics, epoch, decay, time_window, drop_ratio, alpha_times,
        )

    def _build_sample_indicator(self, i, relevant_indices, var_num, dtype):
        """
        为时间戳 i 和给定的 relevant_indices 构造 sample_indicator。
        样本 j 的窗口内，位置 i-j 对应原始时间戳 i。
        """
        n_rel = len(relevant_indices)
        pos_in_window = torch.tensor(
            [i - j for j in relevant_indices],
            device=self.device,
            dtype=torch.long,
        )
        sample_indicator = torch.zeros(
            (n_rel, self.time_window, var_num),
            device=self.device,
            dtype=dtype,
        )
        indices = pos_in_window.view(n_rel, 1, 1).expand(n_rel, 1, var_num)
        sample_indicator.scatter_(1, indices, 1.0)
        return sample_indicator

    def forward(self, origin_x, x, y=None, seq_x_mark=None, seq_y_mark=None, start=None, **kwargs):
        """GTW：逐时间戳更新，每个时间点用相关样本分块算梯度。"""
        self.model.train()

        origin_x = origin_x.float()
        x = x.float()
        if y is not None:
            y = y.float()
        if seq_x_mark is not None:
            seq_x_mark = seq_x_mark.float()
        if seq_y_mark is not None:
            seq_y_mark = seq_y_mark.float()

        if len(origin_x.shape) == 2:
            origin_x = origin_x.unsqueeze(0)

        if start is not None:
            delta = start
        else:
            delta = self.init_delta(origin_x)

        delta = delta.detach()
        data_len = origin_x.size(1)
        sample_num = data_len - self.time_window + 1
        var_num = origin_x.shape[2]
        momentum = [0 for _ in range(data_len)]

        chunk_size = getattr(self.args, 'global_chunk_size', None)
        if chunk_size is None or chunk_size <= 0:
            chunk_size = getattr(self.args, 'batch_size', 64) or 64

        for epoch in range(self.epoch):
            print(f"第 {epoch + 1}/{self.epoch} 轮攻击")
            for i in tqdm(range(data_len), desc="攻击迭代进度"):
                # 直接计算相关样本：样本 j 包含时间戳 i 当且仅当 j <= i < j + time_window
                start_j = max(0, i - self.time_window + 1)
                end_j = min(i + 1, sample_num)
                relevant_indices = list(range(start_j, end_j))
                if not relevant_indices:
                    continue

                keep_num = int(len(relevant_indices) * (1 - self.drop_ratio))
                keep_num = max(1, keep_num)
                perm = torch.randperm(len(relevant_indices))
                relevant_indices = [relevant_indices[idx] for idx in perm[:keep_num]]

                # print(f"相关样本: {relevant_indices}")
                # print(f"drop_ratio: {self.drop_ratio}")
                # print(f"keep_num: {keep_num}")

                # 在 drop/shuffle 之后构造 sample_indicator，保证对齐
                sample_indicator = self._build_sample_indicator(
                    i, relevant_indices, var_num, origin_x.dtype
                )

                delta_i = delta[0, i, :].clone().detach().unsqueeze(0).unsqueeze(0)
                delta_i.requires_grad = True

                grad_sum = torch.zeros_like(delta_i)
                for k in range(0, len(relevant_indices), chunk_size):
                    idx_chunk = relevant_indices[k : k + chunk_size]
                    x_chunk = x[idx_chunk].to(self.device)
                    si_chunk = sample_indicator[k : k + len(idx_chunk)]
                    delta_expand = delta_i.repeat(len(idx_chunk), 1, 1)

                    perturbated_chunk = x_chunk + si_chunk * delta_expand
                    prediction, true = self.get_prediction(
                        self.transform(data=perturbated_chunk, momentum=momentum),
                        y[idx_chunk].to(self.device),
                        seq_x_mark[idx_chunk].to(self.device),
                        seq_y_mark[idx_chunk].to(self.device),
                    )
                    object_value = self.get_object_value(
                        prediction.reshape(-1, 1), true.reshape(-1, 1)
                    )
                    grad_chunk = self.get_grad(object_value, delta_i)
                    grad_sum = grad_sum + grad_chunk

                    del x_chunk, si_chunk, delta_expand, perturbated_chunk, prediction, true, object_value, grad_chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                momentum[i] = self.get_momentum(grad_sum, momentum[i])
                delta_i = self.update_delta(0, momentum[i], self.alpha)
                delta_i = delta_i.detach()
                delta[0, i, :] = delta_i[0, 0, :]

        return delta.detach()[0, :, :]
