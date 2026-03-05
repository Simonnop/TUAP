import copy
import torch
from bayes_opt import BayesianOptimization

from exp.attack.exp_attack_basic import Exp_Attack_Basic
from exp.utils.losses import calcu_loss


def _segment_bounds(seq_len, segments):
    segments = max(1, min(seq_len, segments))
    bounds = []
    for idx in range(segments):
        start = (seq_len * idx) // segments
        end = (seq_len * (idx + 1)) // segments if idx < segments - 1 else seq_len
        if end <= start:
            end = start + 1 if start + 1 <= seq_len else seq_len
        bounds.append((start, end))
    return bounds


def _build_delta_from_config(config, bounds, seq_len, feature_num):
    delta = torch.zeros((seq_len, feature_num), dtype=torch.float32)
    for f_idx in range(feature_num):
        for s_idx, (start, end) in enumerate(bounds):
            key = f"d_{f_idx}_{s_idx}"
            if key in config:
                delta[start:end, f_idx] = config[key]
    return delta


def _predict_with_model(model, x, y, seq_x_mark, seq_y_mark, args, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        batch_size = getattr(args, "predict_batch_size", None)
        if batch_size is None or batch_size <= 0:
            batch_size = getattr(args, "batch_size", 64) or 64
        total_samples = x.shape[0]

        outputs = []
        truths = []

        for start in range(0, total_samples, batch_size):
            end_idx = min(start + batch_size, total_samples)
            batch_x = x[start:end_idx].float().to(device)
            batch_y = y[start:end_idx].float().to(device)
            batch_x_mark = seq_x_mark[start:end_idx].float().to(device)
            batch_y_mark = seq_y_mark[start:end_idx].float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            if getattr(args, "use_amp", False):
                with torch.cuda.amp.autocast():
                    outputs_i = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs_i = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs_i = outputs_i[:, -args.pred_len:, :].cpu()
            batch_y = batch_y[:, -args.pred_len:, :].cpu()
            outputs_i = outputs_i[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:]

            outputs.append(outputs_i)
            truths.append(batch_y)

        outputs = torch.cat(outputs, dim=0)
        truths = torch.cat(truths, dim=0)
    return outputs, truths


def _bo_objective_internal(config, context, segment_bounds):
    delta_feat = _build_delta_from_config(config, segment_bounds, context["seq_len"], context["feature_num"])
    
    # context["all_x"] is likely CPU tensor. delta_feat is CPU tensor (default creation).
    # Perform addition on CPU to avoid potential OOM if all_x is large
    attacked_x = context["all_x"] + delta_feat.unsqueeze(0)
    
    predictions, truths = _predict_with_model(
        context["victim_model"],
        attacked_x,
        context["all_y"],
        context["all_x_mark"],
        context["all_y_mark"],
        context["args"],
        context["device"]
    )
    loss_value = calcu_loss(predictions, context["all_true"])[context["metrics"]]
    return float(loss_value)


class Exp_Bo_Method(Exp_Attack_Basic):

    def __init__(self, args):
        super(Exp_Bo_Method, self).__init__(args)

    def attack(self):
        all_x, all_y, all_x_mark, all_y_mark, _, all_true = self._get_sample_input()

        segments = getattr(self.args, "bo_segments", 12)
        bounds = _segment_bounds(self.time_window, segments)
        
        # Define bounds for BayesianOptimization
        # Key format: d_{feature_idx}_{segment_idx}
        pbounds = {
            f"d_{f_idx}_{s_idx}": (-self.epsilon, self.epsilon)
            for f_idx in range(self.feature_num)
            for s_idx in range(len(bounds))
        }

        # Context for the objective function - Global Context
        context = {
            "victim_model": self.victim_model,
            "all_x": all_x,
            "all_y": all_y,
            "all_x_mark": all_x_mark,
            "all_y_mark": all_y_mark,
            "all_true": all_true,
            "args": self.args,
            "device": self.device,
            "metrics": self.metrics,
            "feature_num": self.feature_num,
            "seq_len": self.time_window,
        }

        def objective_function(**config):
            return _bo_objective_internal(config, context, bounds)

        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=pbounds,
            random_state=1,
            verbose=0,
            allow_duplicate_points=True
        )

        num_trials = getattr(self.args, "bo_trials", 20)
        # Allocate trials between random exploration and optimization
        # Use roughly 25% for initialization, but at least 2
        init_points = max(2, int(num_trials * 0.25))
        n_iter = max(1, num_trials - init_points)

        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )

        best_config = optimizer.max['params']
        
        best_delta_feat = _build_delta_from_config(best_config, bounds, self.time_window, self.feature_num)
        
        # Repeat for all samples to match shape
        all_deltas = best_delta_feat.unsqueeze(0).repeat(all_x.size(0), 1, 1)

        self.convert_to_global(all_deltas, all_x, all_y, all_x_mark, all_y_mark, all_true)
