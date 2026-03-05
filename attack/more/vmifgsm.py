import torch
from ..base_attack import BaseAttack
from ..global_attack import GlobalGradientAccumulationAttack
from tqdm import tqdm

class VMIFGSM(BaseAttack):
    """
    VMI-FGSM Attack (Window-wise)
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mse', epoch=10, decay=1, alpha_times=1, beta=1.5, num_neighbor=20):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, alpha_times)
        self.beta = beta
        self.num_neighbor = num_neighbor
        self.radius = beta * epsilon

    def get_variance(self, data, delta, y, seq_x_mark, seq_y_mark, cur_grad, momentum):
        grad_sum = 0
        for _ in range(self.num_neighbor):
            # Add random noise for variance estimation
            noise = torch.zeros_like(delta).uniform_(-self.radius, self.radius).to(self.device)
            # Apply transform (with momentum if needed, though VMI usually uses Nesterov too? 
            # The CV implementation uses self.transform which might be just identity or Nesterov if mixed.
            # Here we assume standard VMI-FGSM doesn't strictly imply Nesterov unless specified, 
            # but CV implementation DOES use self.transform(..., momentum=momentum). 
            # BaseAttack.transform is identity. If we want VMI-NIFGSM we should mix them.
            # For now, we use BaseAttack.transform which is identity.
            
            perturbated_data = delta + data + noise
            prediction, true = self.get_prediction(self.transform(data=perturbated_data, momentum=momentum), y, seq_x_mark, seq_y_mark)
            object_value = self.get_object_value(prediction, true)
            grad = self.get_grad(object_value, delta)
            grad_sum += grad
            
        return grad_sum / self.num_neighbor - cur_grad

    def forward(self, data, y=None, seq_x_mark=None, seq_y_mark=None, **kwargs):
        self.model.train()

        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        data = data.float().to(self.device)
        delta = self.init_delta(data)

        momentum = 0
        variance = 0
        
        for _ in range(self.epoch):
            perturbated_data = delta + data
            prediction, true = self.get_prediction(self.transform(data=perturbated_data, momentum=momentum), y, seq_x_mark, seq_y_mark)
            object_value = self.get_object_value(prediction, true)
            grad = self.get_grad(object_value, delta)

            # Update momentum with variance
            momentum = self.get_momentum(grad + variance, momentum)
            
            # Calculate variance for next step
            variance = self.get_variance(data, delta, y, seq_x_mark, seq_y_mark, grad, momentum)

            delta = self.update_delta(delta, momentum, self.alpha)

        delta, true = delta.detach(), true.detach()
        return delta, true

class GGAA_VMIFGSM(GlobalGradientAccumulationAttack):
    """
    GGAA with VMI-FGSM (Global)
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mae', epoch=10, decay=1, time_window=None, drop_ratio=0., alpha_times=1, beta=1.5, num_neighbor=5):
        # Reduced num_neighbor default for Global attack efficiency
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, time_window, drop_ratio, alpha_times)
        self.beta = beta
        self.num_neighbor = num_neighbor
        self.radius = beta * epsilon

    def forward(self, origin_x, x, y=None, seq_x_mark=None, seq_y_mark=None, **kwargs):
        if y is None or seq_x_mark is None or seq_y_mark is None:
            raise ValueError("GGAA_VMIFGSM attack requires y, seq_x_mark, and seq_y_mark")

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
        variance = torch.zeros_like(delta)

        for epoch in range(self.epoch):
            global_grad = torch.zeros_like(delta)
            global_variance = torch.zeros_like(delta) # We accumulate variance too? 
            # Or we compute variance locally?
            # VMI: v = E[g(x+d+r)] - g(x+d)
            # In GGAA, we sum gradients.
            # Let's compute global_grad normally first.
            
            print(f"Attack Epoch {epoch + 1}/{self.epoch} (GGAA_VMIFGSM)")
            
            for idx in tqdm(range(sample_num), desc="Attack Progress"):
                start = idx
                end = start + window_len
                
                delta_slice = delta[:, start:end, :]
                
                # 1. Compute Gradient on clean (perturbed) input
                perturbed_window = origin_x[:, start:end, :] + delta_slice
                y_slice = y[idx:idx + 1].to(self.device)
                x_mark_slice = seq_x_mark[idx:idx + 1].to(self.device)
                y_mark_slice = seq_y_mark[idx:idx + 1].to(self.device)

                prediction, true = self.get_prediction(perturbed_window, y_slice, x_mark_slice, y_mark_slice)
                loss = self.get_object_value(prediction, true)
                grad = torch.autograd.grad(loss, delta_slice, retain_graph=False, create_graph=False)[0]
                global_grad[:, start:end, :] += grad.detach()

                # 2. Compute Variance (Neighbor gradients)
                # To be efficient, maybe we only sample neighbors for the current window?
                # V = Avg(Grad(neighbors)) - Grad(current)
                neighbor_grad_sum = torch.zeros_like(grad)
                for _ in range(self.num_neighbor):
                    noise = torch.zeros_like(delta_slice).uniform_(-self.radius, self.radius).to(self.device)
                    perturbed_window_noise = origin_x[:, start:end, :] + delta_slice + noise
                    
                    pred_n, true_n = self.get_prediction(perturbed_window_noise, y_slice, x_mark_slice, y_mark_slice)
                    loss_n = self.get_object_value(pred_n, true_n)
                    grad_n = torch.autograd.grad(loss_n, delta_slice, retain_graph=False, create_graph=False)[0]
                    neighbor_grad_sum += grad_n.detach()
                
                local_variance = (neighbor_grad_sum / self.num_neighbor) - grad.detach()
                global_variance[:, start:end, :] += local_variance

                del perturbed_window, y_slice, x_mark_slice, y_mark_slice, prediction, true, loss, grad

            # Normalize global gradient
            norm = global_grad.abs().sum()
            g_norm = global_grad / (norm + 1e-12)
            
            # Normalize global variance? 
            # CV VMI just adds variance to gradient before momentum.
            # Here g_norm is normalized. Variance should probably be normalized similarly or added before normalization?
            # VMI paper: g_new = g + v. m = mu * m + g_new.
            # Here global_grad is the "g". global_variance is "v".
            
            # But global_grad is huge (summed).
            # We usually normalize g by L1 norm in this code.
            # Should we add variance before normalization?
            # g_total = global_grad + global_variance.
            
            # g_total = global_grad + global_variance
            # g_total_norm = g_total.abs().sum()
            # g_final = g_total / (g_total_norm + 1e-12)
            
            # Update momentum
            # Note: The implementation of VMIFGSM uses 'variance' from PREVIOUS step in momentum update?
            # CV code: momentum = get_momentum(grad + variance, momentum); variance = get_variance(...)
            # The 'variance' used in get_momentum is from the PREVIOUS loop iteration.
            # My Window-wise implementation followed this.
            # Here, I computed `global_variance` for CURRENT step.
            # If I want to follow CV exactly, I should use `variance` (stored) and update it.
            # But in the first step variance is 0.
            
            # Let's use the stored variance.
            # But wait, in CV code:
            # momentum = get_momentum(grad + variance, momentum) -> uses previous variance.
            # variance = get_variance(...) -> computes new variance.
            
            # So I should add `variance` (from self) to `global_grad`.
            # But `global_grad` is unnormalized.
            # `variance` should be on the same scale.
            
            # Let's refine:
            # 1. global_grad computed.
            # 2. Add previous variance?
            # In CV: grad (local) + variance (local).
            # Here: global_grad (summed).
            # The variance should be summed too?
            # Let's assume `variance` variable stores the global variance from previous epoch.
            
            # Correct logic:
            # momentum = self.decay * momentum + Normalize(global_grad + variance)
            # variance = global_variance (computed this step)
            
            # Wait, `get_momentum` in `BaseAttack` does: `momentum * decay + grad / mean(abs(grad))`.
            # In `GlobalGradientAccumulationAttack`, it does: `decay * momentum + global_grad / sum(abs(global_grad))`.
            
            # I will follow `GlobalGradientAccumulationAttack` style.
            
            g_with_var = global_grad + variance
            norm_val = g_with_var.abs().sum()
            g_norm = g_with_var / (norm_val + 1e-12)
            
            momentum = self.decay * momentum + g_norm
            
            # Update variance for next step
            variance = global_variance # This is the variance computed in this step
            
            update = delta + self.alpha * momentum.sign()
            delta = torch.clamp(update, -self.epsilon, self.epsilon).detach().requires_grad_(True)
            momentum = momentum.detach()
            variance = variance.detach()

        return delta.detach()[0]
