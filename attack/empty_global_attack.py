import torch


class EmptyGlobalAttack:
    """
    空的全局攻击器：用于 Exp_Global_Method 的固定开销基线。

    - 接口与 GlobalTimestampWise / BaseGlobalAttack 一致（forward/ __call__）
    - 不做模型前向/反向，只返回全零扰动
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
        time_window=None,
        drop_ratio=0.0,
        alpha_times=1,
        random_start=False,
    ):
        self.attack = attack
        self.model = model
        self.args = args
        self.device = next(self.model.parameters()).device if device is None else device

    def forward(self, origin_x, x=None, y=None, seq_x_mark=None, seq_y_mark=None, start=None, **kwargs):
        origin_x = origin_x.float()
        if len(origin_x.shape) == 2:
            origin_x = origin_x.unsqueeze(0)
        origin_x = origin_x.to(self.device)

        delta = torch.zeros_like(origin_x)
        # 与 GTW 一致：返回 (T, F)
        return delta.detach()[0, :, :]

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

