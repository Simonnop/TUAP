import torch
from ..base_attack import BaseAttack
from ..global_attack import GlobalGradientAccumulationAttack
from tqdm import tqdm

class IEFGSM(BaseAttack):
    """
    IE-FGSM Attack (Window-wise)
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mse', epoch=10, alpha_times=1, decay=1.0):
        super().__init__(attack, model, epsilon, norm, device, args, metrics)
        self.epoch = epoch
        self.alpha = self.epsilon / self.epoch * alpha_times
        self.decay = decay

    def forward(self, data, label, seq_x_mark=None, seq_y_mark=None, **kwargs):
        # Window-wise forward
        # data: (B, T, C)
        
        data = data.clone().detach().to(self.device)
        delta = self.init_delta(data)
        momentum = 0
        
        for _ in range(self.epoch):
            # 1. Present Gradient
            prediction, true = self.get_prediction(data + delta, label, seq_x_mark, seq_y_mark)
            loss = self.get_object_value(prediction, true)
            grad = self.get_grad(loss, delta)
            
            # Normalize
            g_p = grad / (grad.abs().mean(dim=(1,2), keepdim=True) + 1e-12)
            
            # 2. Anticipatory Gradient
            # Lookahead point
            delta_a = delta + self.alpha * g_p
            
            prediction_a, true_a = self.get_prediction(data + delta_a, label, seq_x_mark, seq_y_mark)
            loss_a = self.get_object_value(prediction_a, true_a)
            grad_a = self.get_grad(loss_a, delta) # Gradient w.r.t delta at delta_a point (approx)
            # Actually we want grad w.r.t input at x+delta_a. 
            # get_grad usually computes d(loss)/d(delta). 
            # Since delta_a is derived from delta, we need to be careful.
            # Usually implementation is: forward(x+delta_a), backward(), get x.grad.
            # In BaseAttack, get_grad does torch.autograd.grad(loss, delta).
            # If we pass delta_a to prediction, loss depends on delta_a.
            # But we want gradient w.r.t delta? Or just gradient at that point?
            # CV implementation: 
            # logits = get_logits(transform(data+delta+alpha*g_p))
            # loss = ...
            # grad = get_grad(loss, delta) 
            # Wait, if loss is computed from (data+delta+...), but we ask grad wrt delta, it works if delta is in graph.
            # But delta_a = delta + ... so it is in graph.
            # However, typically we just want the gradient vector at that location.
            # BaseAttack.get_grad takes (loss, delta). 
            
            g_a = grad_a / (grad_a.abs().mean(dim=(1,2), keepdim=True) + 1e-12)
            
            # Update momentum
            momentum = self.decay * momentum + (g_p + g_a) / 2
            
            # Update delta
            delta = self.update_delta(delta, momentum, self.alpha)
            
        return delta.detach(), true


class GGAA_IEFGSM(GlobalGradientAccumulationAttack):
    """
    GGAA with IE-FGSM (Global)
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mae', epoch=10, decay=1.0, time_window=None, drop_ratio=0., alpha_times=1):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, time_window, drop_ratio)
        self.alpha = self.epsilon / self.epoch * alpha_times
        self.decay = decay

    def forward(self, origin_x, x, y=None, seq_x_mark=None, seq_y_mark=None, **kwargs):
        if y is None or seq_x_mark is None or seq_y_mark is None:
            raise ValueError("GGAA_IEFGSM attack requires y, seq_x_mark, and seq_y_mark")

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
        
        for epoch in range(self.epoch):
            print(f"Attack Epoch {epoch + 1}/{self.epoch} (GGAA_IEFGSM)")
            
            # --- Pass 1: Present Gradient (g_p) ---
            global_grad_p = torch.zeros_like(delta)
            
            # Using loop for memory efficiency (GGAA standard)
            for idx in tqdm(range(sample_num), desc="Pass 1 (Present Grad)"):
                start = idx
                end = start + window_len
                
                delta_slice = delta[:, start:end, :]
                y_slice = y[idx:idx + 1].to(self.device)
                x_mark_slice = seq_x_mark[idx:idx + 1].to(self.device)
                y_mark_slice = seq_y_mark[idx:idx + 1].to(self.device)
                
                prediction, true = self.get_prediction(origin_x[:, start:end, :] + delta_slice, y_slice, x_mark_slice, y_mark_slice)
                loss = self.get_object_value(prediction, true)
                
                g = torch.autograd.grad(loss, delta_slice, retain_graph=False, create_graph=False)[0]
                global_grad_p[:, start:end, :] += g.detach()
            
            # Normalize g_p
            # Note: Normalization strategy in CV was mean(abs). 
            # In GGAA standard, we usually just sum or use sign. 
            # But IEFGSM relies on this specific normalization.
            # Global normalization:
            g_p = global_grad_p / (global_grad_p.abs().mean() + 1e-12) 
            # Note: CV code used mean(dim=(1,2,3)). Here global_grad_p is (1, L, C). Mean over all is fine.
            
            
            # --- Pass 2: Anticipatory Gradient (g_a) ---
            global_grad_a = torch.zeros_like(delta)
            
            # Temporary lookahead delta
            delta_temp = delta + self.alpha * g_p
            
            for idx in tqdm(range(sample_num), desc="Pass 2 (Anticipatory Grad)"):
                start = idx
                end = start + window_len
                
                delta_slice = delta_temp[:, start:end, :]
                # We need gradient w.r.t the input at this new point.
                # Since delta_temp is detached (or we treat it as leaf for grad calculation),
                # we want d(Loss)/d(Input) at X + delta_temp.
                
                # To do this cleanly without graph issues:
                delta_slice_var = delta_slice.clone().detach().requires_grad_(True)
                
                y_slice = y[idx:idx + 1].to(self.device)
                x_mark_slice = seq_x_mark[idx:idx + 1].to(self.device)
                y_mark_slice = seq_y_mark[idx:idx + 1].to(self.device)
                
                prediction, true = self.get_prediction(origin_x[:, start:end, :] + delta_slice_var, y_slice, x_mark_slice, y_mark_slice)
                loss = self.get_object_value(prediction, true)
                
                g = torch.autograd.grad(loss, delta_slice_var, retain_graph=False, create_graph=False)[0]
                global_grad_a[:, start:end, :] += g.detach()

            # Normalize g_a
            g_a = global_grad_a / (global_grad_a.abs().mean() + 1e-12)

            # Update momentum
            momentum = self.decay * momentum + (g_p + g_a) / 2
            
            # Update delta
            # Standard FGSM/I-FGSM update uses sign. IEFGSM usually uses the raw momentum value (scaled)?
            # CV code: delta = update_delta(delta, ..., momentum, alpha)
            # update_delta in BaseAttack usually uses sign() for linfty.
            # Let's check IEFGSM CV code again. 
            # It inherits Attack. update_delta defaults to sign() if norm is linfty.
            # So yes, we should use sign() of momentum.
            
            update = delta + self.alpha * momentum.sign()
            delta = torch.clamp(update, -self.epsilon, self.epsilon).detach().requires_grad_(True)
            momentum = momentum.detach()

        return delta.detach()[0]
