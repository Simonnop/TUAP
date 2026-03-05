import torch
from ..base_attack import BaseAttack
from ..global_attack import GlobalGradientAccumulationAttack
from tqdm import tqdm

class NIFGSM(BaseAttack):
    """
    NI-FGSM Attack (Window-wise)
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mse', epoch=10, decay=1, alpha_times=1):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, alpha_times)
        
    def transform(self, data, momentum=None, **kwargs):
        """
        Apply Nesterov momentum lookahead
        """
        if momentum is None:
            return data
        return data + self.alpha * self.decay * momentum

class GGAA_NIFGSM(GlobalGradientAccumulationAttack):
    """
    GGAA with NI-FGSM (Global)
    """
    def __init__(self, attack, model, epsilon, norm, device=None, args=None, metrics='mae', epoch=10, decay=1, time_window=None, drop_ratio=0., alpha_times=1):
        super().__init__(attack, model, epsilon, norm, device, args, metrics, epoch, decay, time_window, drop_ratio, alpha_times)

    def forward(self, origin_x, x, y=None, seq_x_mark=None, seq_y_mark=None, **kwargs):
        if y is None or seq_x_mark is None or seq_y_mark is None:
            raise ValueError("GGAA_NIFGSM attack requires y, seq_x_mark, and seq_y_mark")

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
            global_grad = torch.zeros_like(delta)
            # 调试：当 epoch>=1 时 lookahead 应非零，否则与 GGAA 等价
            lookahead_norm = (self.alpha * self.decay * momentum).abs().sum().item()
            if epoch >= 1 and lookahead_norm < 1e-10:
                import warnings
                warnings.warn(f"GGAA_NIFGSM: epoch={epoch+1} lookahead≈0 (norm={lookahead_norm}), 与 GGAA 等价。请检查 decay={self.decay}")
            print(f"Attack Epoch {epoch + 1}/{self.epoch} (GGAA_NIFGSM)")
            
            for idx in tqdm(range(sample_num), desc="Attack Progress"):
                start = idx
                end = start + window_len
                
                # Nesterov lookahead: add momentum term to the input
                # delta_slice corresponds to delta at [start:end]
                # momentum_slice corresponds to momentum at [start:end]
                delta_slice = delta[:, start:end, :]
                momentum_slice = momentum[:, start:end, :]
                
                # Apply lookahead
                lookahead_term = self.alpha * self.decay * momentum_slice
                perturbed_window = origin_x[:, start:end, :] + delta_slice + lookahead_term

                y_slice = y[idx:idx + 1].to(self.device)
                x_mark_slice = seq_x_mark[idx:idx + 1].to(self.device)
                y_mark_slice = seq_y_mark[idx:idx + 1].to(self.device)

                prediction, true = self.get_prediction(perturbed_window, y_slice, x_mark_slice, y_mark_slice)
                loss = self.get_object_value(prediction, true)
                
                # Compute gradient w.r.t delta_slice (which is part of the graph via perturbed_window)
                # Note: we need to be careful. perturbed_window = constant + delta_slice + constant_momentum
                # So grad w.r.t delta_slice is correct.
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
