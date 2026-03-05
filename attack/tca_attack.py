import torch
import torch.nn as nn
from .base_attack import BaseAttack

class TCAAttack(BaseAttack):
    """
    Temporal Characteristics-based Adversarial Attack (TCA)
    Paper: Shen and Li - 2025 - Temporal characteristics-based adversarial attacks on time series forecasting.pdf
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mae', epoch=10, decay=1, alpha_times=1, lambda_mean=0.1, lambda_std=0.1, lambda_trend=0.1):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, alpha_times)
        self.lambda_mean = lambda_mean
        self.lambda_std = lambda_std
        self.lambda_trend = lambda_trend

    def forward(self, data, y=None, seq_x_mark=None, seq_y_mark=None, **kwargs):
        """
        TCA forward process with temporal constraints
        """
        self.model.train()

        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        data = data.float().to(self.device)
        
        # Initialize delta
        delta = self.init_delta(data)
        
        momentum = 0
        for _ in range(self.epoch):
            delta.requires_grad_(True)
            perturbated_data = data + delta
            
            # 1. Forecasting Loss (to be maximized)
            prediction, true = self.get_prediction(self.transform(data=perturbated_data, momentum=momentum), y, seq_x_mark, seq_y_mark)
            forecasting_loss = self.get_object_value(prediction, true)
            
            # 2. Temporal Characteristic Loss (to be minimized)
            # Mean constraint
            loss_mean = torch.mean(torch.abs(torch.mean(perturbated_data, dim=1) - torch.mean(data, dim=1)))
            
            # Std constraint
            loss_std = torch.mean(torch.abs(torch.std(perturbated_data, dim=1) - torch.std(data, dim=1)))
            
            # Trend constraint (First-order difference)
            diff_orig = data[:, 1:, :] - data[:, :-1, :]
            diff_adv = perturbated_data[:, 1:, :] - perturbated_data[:, :-1, :]
            loss_trend = torch.mean(torch.abs(diff_adv - diff_orig))
            
            # Total Objective: Maximize (Forecasting Loss - Lambda * Temporal Loss)
            # Since we usually use gradient ascent for maximization, or minimize negative objective.
            # BaseAttack's update_delta uses grad directly, which usually assumes gradient ascent for untargeted attacks.
            
            total_objective = forecasting_loss - (self.lambda_mean * loss_mean + self.lambda_std * loss_std + self.lambda_trend * loss_trend)
            
            # Calculate gradient
            grad = torch.autograd.grad(total_objective, delta)[0]
            
            # Update momentum
            momentum = self.get_momentum(grad, momentum)
            
            # Update delta
            delta = self.update_delta(delta, momentum, self.alpha)
            
        return delta.detach(), true.detach()

    def get_object_value(self, prediction: torch.Tensor, y=None):
        """
        Override to return the specified metric loss
        """
        losses = self.calcu_loss(prediction, y)
        return losses[self.metrics]

