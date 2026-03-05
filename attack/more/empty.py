import torch


class EMPTY:
    """
    空攻击：用于测量流程固定开销（取样/拼 batch/搬运/回传/清缓存等）。

    约定：
    - 不做模型前向/反向，不计算梯度
    - 仅做与真实攻击相同的张量搬运与形状对齐，并返回全零 delta
    """

    def __init__(
        self,
        attack,
        model,
        epsilon,
        norm,
        device=None,
        args=None,
        metrics="mse",
        epoch=1,
        decay=0,
        alpha_times=1,
    ):
        self.attack = attack
        self.model = model
        self.args = args
        self.device = next(self.model.parameters()).device if device is None else device

    def __call__(self, data, y=None, seq_x_mark=None, seq_y_mark=None, **kwargs):
        # 对齐 BaseAttack.forward 的输入维度习惯
        if len(data.shape) == 2:
            data = data.unsqueeze(0)

        # 模拟真实攻击的搬运与分配（但不做 forward/backward）
        data = data.float().to(self.device)
        delta = torch.zeros_like(data)

        # 返回一个与真实攻击一致的 true 形状（仅做切片/搬运，不做模型预测）
        if y is None:
            true = torch.empty((data.shape[0], 0), device=self.device)
        else:
            batch_y = y.float().to(self.device)
            true = batch_y[:, -self.args.pred_len :, :]
            f_dim = -1 if self.args.features == "MS" else 0
            true = true[:, :, f_dim:]

        return delta.detach(), true.detach()

