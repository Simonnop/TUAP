import torch
from ..base_attack import BaseAttack
from ..global_attack import GlobalGradientAccumulationAttack

class FGSM(BaseAttack):
    """
    FGSM Attack (Window-wise)
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mse', epoch=1, decay=0, alpha_times=1):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, alpha_times)

class GGAA_FGSM(GlobalGradientAccumulationAttack):
    """
    GGAA with FGSM (Global)
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mae', epoch=1, decay=0, time_window=None, drop_ratio=0, alpha_times=1):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, time_window, drop_ratio, alpha_times)
