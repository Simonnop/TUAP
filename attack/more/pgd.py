import torch
from ..base_attack import BaseAttack
from ..global_attack import GlobalGradientAccumulationAttack

class PGD(BaseAttack):
    """
    PGD Attack (Window-wise)
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mse', epoch=10, decay=0, alpha_times=1):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, alpha_times)

class GGAA_PGD(GlobalGradientAccumulationAttack):
    """
    GGAA with PGD (Global)
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mae', epoch=10, decay=0, time_window=None, drop_ratio=0.5, alpha_times=1, random_start=True):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, time_window, drop_ratio, alpha_times, random_start)
