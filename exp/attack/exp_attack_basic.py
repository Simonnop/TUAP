import numpy as np
import torch
from torch import device
import os
from tqdm import tqdm
import joblib
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from multiprocessing import Pool

from data_provider.data_factory import data_provider
from tools.record import update_record,load_solution
from exp.utils.losses import calcu_loss

import numpy as np
from mealpy import IntegerVar, Problem
from mealpy.bio_based import SMA
from exp.utils.ts_tool import TimeSeriesTool
from exp.utils.attacker import Attacker

class Exp_Attack_Basic:
    def __init__(self, args):
        # 
        self.args = args
        self.seed = args.seed
        self.kind = args.kind
        self.norm = "linfty"
        self.metrics = 'mae' if args.attack_algo == 'ATSG' else 'mse'
        print(f"metrics: {self.metrics}")
        
        self.attack_algo = args.attack_algo

        self.epsilon = args.epsilon
        self.attack_rate = args.attack_rate
        self.time_window = args.seq_len
        self.feature_num = args.enc_in
        self.flag = 'test' if not hasattr(args, 'flag') or args.flag is None else args.flag

        self.device = self._acquire_device()
        self.generate_model, self.victim_model = self._build_model()
        self.data_set, self.data_loader = self._get_data(flag=self.flag)
        self.attacker = self._get_attacker()

        # 拿数据尺寸
        self.sample_num = len(self.data_set)
        self.len = self.sample_num + self.time_window - 1
        self.space = self.time_window

    def _get_data(self, flag="test"):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_attacker(self):
        attacker = Attacker(self.args, generate_model=self.generate_model, metrics=self.metrics, device=self.device).get_attacker(epsilon=self.epsilon, global_=False)
        return attacker
        
    def _acquire_device(self):
        if self.args.device == 'cuda':
            device_id = np.random.randint(torch.cuda.device_count())
            device = torch.device('cuda:{}'.format(device_id))
        else:
            device = torch.device(self.args.device)
        return device

    def _build_model(self):
        # 加载模型
        # 尝试加载带 seed 的模型，如果不存在则尝试加载不带 seed 的模型
        
        # 针对生成模型（代理模型）的路径查找
        if "defense" in self.args.generate_model:
            gen_model_path_with_seed = os.path.join("./checkpoints/", f"{self.args.generate_model}_s{self.args.seed}", "checkpoint.pth")
            gen_model_path_no_seed = os.path.join("./checkpoints/", f"{self.args.generate_model}", "checkpoint.pth")
        else:
            gen_model_path_with_seed = os.path.join("./checkpoints/" + f"{self.args.data}_{self.args.seq_len}_{self.args.pred_len}_{self.args.generate_model}_s{self.args.seed}", "checkpoint.pth")
            gen_model_path_no_seed = os.path.join("./checkpoints/" + f"{self.args.data}_{self.args.seq_len}_{self.args.pred_len}_{self.args.generate_model}", "checkpoint.pth")
        
        def _torch_load_compat(path):
            # PyTorch>=2.6 默认 weights_only=True，旧工程保存的是整模型对象时会加载失败
            try:
                return torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=self.device)

        if os.path.exists(gen_model_path_with_seed):
            print(f"Loading generate model from {gen_model_path_with_seed}")
            generate_model = _torch_load_compat(gen_model_path_with_seed)
        elif os.path.exists(gen_model_path_no_seed):
            print(f"Loading generate model from {gen_model_path_no_seed}")
            generate_model = _torch_load_compat(gen_model_path_no_seed)
        else:
            # 最后的备选方案：如果都没有，打印报错路径
            print(f"Error: Could not find generate model at {gen_model_path_with_seed} or {gen_model_path_no_seed}")
            raise FileNotFoundError(f"Model not found: {self.args.generate_model}")

        # 针对受害模型（目标模型）的路径查找
        if "defense" in self.args.victim_model:
            vic_model_path_with_seed = os.path.join("./checkpoints/", f"{self.args.victim_model}_s{self.args.seed}", "checkpoint.pth")
            vic_model_path_no_seed = os.path.join("./checkpoints/", f"{self.args.victim_model}", "checkpoint.pth")
        else:
            vic_model_path_with_seed = os.path.join("./checkpoints/" + f"{self.args.data}_{self.args.seq_len}_{self.args.pred_len}_{self.args.victim_model}_s{self.args.seed}", "checkpoint.pth")
            vic_model_path_no_seed = os.path.join("./checkpoints/" + f"{self.args.data}_{self.args.seq_len}_{self.args.pred_len}_{self.args.victim_model}", "checkpoint.pth")

        if os.path.exists(vic_model_path_with_seed):
            print(f"Loading victim model from {vic_model_path_with_seed}")
            victim_model = _torch_load_compat(vic_model_path_with_seed)
        elif os.path.exists(vic_model_path_no_seed):
            print(f"Loading victim model from {vic_model_path_no_seed}")
            victim_model = _torch_load_compat(vic_model_path_no_seed)
        else:
            print(f"Error: Could not find victim model at {vic_model_path_with_seed} or {vic_model_path_no_seed}")
            raise FileNotFoundError(f"Model not found: {self.args.victim_model}")

        # 攻击阶段不需要更新参数：切换为 eval 并冻结参数，降低显存占用
        generate_model.train()
        victim_model.train()
        for p in generate_model.parameters():
            p.requires_grad_(False)
        for p in victim_model.parameters():
            p.requires_grad_(False)

        # TODO 检查 RNN 下使用 Train 的问题
        
        return generate_model, victim_model
    
    def attack(self):
        raise NotImplementedError("子类必须实现此方法")
    
    def load_attack(self): 
        # 如果存在样本delta, 则加载样本delta
        if 'GGAA' or 'GTW' in self.attack_algo:
            if self.check_delta_exist('global'):
                print("Loading sample delta")
                all_x, all_y, all_x_mark, all_y_mark, all_deltas, all_true = self._get_sample_input()
                all_deltas = self.load_sample_delta('global')
                all_prediction, _ = self.get_prediction(self.victim_model, all_x + all_deltas, all_y, all_x_mark, all_y_mark)
                self.cal_loss(all_prediction, all_true)
                return
            else:
                print("No existing delta, Generating sample delta")
                self.attack()
                return

        if self.check_delta_exist('random') and self.check_delta_exist('first') and self.check_delta_exist('last') and self.check_delta_exist('raw'):
            print("Loading sample delta")
            # 获取样本
            all_x, all_y, all_x_mark, all_y_mark, all_deltas, all_true = self._get_sample_input()
            # raw
            self.kind = 'raw'
            self.args.kind = 'raw'
            all_deltas = self.load_sample_delta('raw')
            all_prediction, _ = self.get_prediction(self.victim_model, all_x + all_deltas, all_y, all_x_mark, all_y_mark)
            self.cal_loss(all_prediction, all_true)
            # first
            self.kind = 'first'
            self.args.kind = 'first'
            all_deltas = self.load_sample_delta('first')
            all_prediction, _ = self.get_prediction(self.victim_model, all_x + all_deltas, all_y, all_x_mark, all_y_mark)
            self.cal_loss(all_prediction, all_true)
            # last
            self.kind = 'last'
            self.args.kind = 'last'
            all_deltas = self.load_sample_delta('last')
            all_prediction, _ = self.get_prediction(self.victim_model, all_x + all_deltas, all_y, all_x_mark, all_y_mark)
            self.cal_loss(all_prediction, all_true)
            # random
            self.kind = 'random'
            self.args.kind = 'random'
            all_deltas = self.load_sample_delta('random')
            all_prediction, _ = self.get_prediction(self.victim_model, all_x + all_deltas, all_y, all_x_mark, all_y_mark)
            self.cal_loss(all_prediction, all_true)
        else:
            print("No existing delta, Generating sample delta")
            self.attack()
    
    def _get_dataset_items(self):
        all_x = []
        all_y = []
        all_x_mark = []
        all_y_mark = []

        for i in tqdm(range(len(self.data_set)), desc="获取数据样本"):
            batch_x, batch_y, seq_x_mark, seq_y_mark = self.data_set[i]
            # 将NumPy数组转换为PyTorch张量
            all_x.append(torch.as_tensor(batch_x, device=self.device))
            all_y.append(torch.as_tensor(batch_y, device=self.device))
            all_x_mark.append(torch.as_tensor(seq_x_mark, device=self.device))
            all_y_mark.append(torch.as_tensor(seq_y_mark, device=self.device))
        
        all_x = torch.stack(all_x)
        all_y = torch.stack(all_y)
        all_x_mark = torch.stack(all_x_mark)
        all_y_mark = torch.stack(all_y_mark)

        return all_x, all_y, all_x_mark, all_y_mark
    
    def _get_sample_delta(self, by_direction=False):
        # 获取样本 delta 后, 对每个位置进行启发式选择, 选择最佳 delta
        # 正负方向攻击, 得到 deltas
        batch_x, batch_y, seq_x_mark, seq_y_mark = self.data_set[0]
            
        # 检查并打印数据形状
        print(f"单个样本形状: batch_x: {batch_x.shape}, batch_y: {batch_y.shape}")
        print(f"单个样本形状: seq_x_mark: {seq_x_mark.shape}, seq_y_mark: {seq_y_mark.shape}")
        
        all_x = []
        all_y = []
        all_x_mark = []
        all_y_mark = []
        all_deltas = []
        all_true = []
        # 保存攻击方向信息
        all_directions = []
        
        # 设置批处理大小：与全局约定一致，统一从 args.batch_size 读取
        batch_size = getattr(self.args, 'batch_size', None)
        if batch_size is None or batch_size <= 0:
            batch_size = 32
        
        # 分批处理数据集
        for i in tqdm(range(0, len(self.data_set), batch_size), desc="进行对抗攻击"):
            # 获取当前批次的索引范围
            batch_indices = range(i, min(i + batch_size, len(self.data_set)))
            
            # 为当前批次创建张量
            batch_all_x = []
            batch_all_y = []
            batch_all_x_mark = []
            batch_all_y_mark = []
            
            # 收集当前批次的数据（保持在 CPU，计算时再搬到 GPU）
            for j in batch_indices:
                batch_x, batch_y, seq_x_mark, seq_y_mark = self.data_set[j]
                batch_all_x.append(torch.as_tensor(batch_x))
                batch_all_y.append(torch.as_tensor(batch_y))
                batch_all_x_mark.append(torch.as_tensor(seq_x_mark))
                batch_all_y_mark.append(torch.as_tensor(seq_y_mark))
            
            # 将列表转换为张量
            batch_all_x = torch.stack(batch_all_x)
            batch_all_y = torch.stack(batch_all_y)
            batch_all_x_mark = torch.stack(batch_all_x_mark)
            batch_all_y_mark = torch.stack(batch_all_y_mark)
            
            # 打印当前批次的形状
            # print(f"批次 {i//batch_size+1} 数据形状: batch_all_x: {batch_all_x.shape}")
            
            # 对当前批次计算delta
            if by_direction == False:
                batch_deltas, batch_true = self.attacker(batch_all_x, batch_all_y, batch_all_x_mark, batch_all_y_mark)
            else:
                direction_attacker = Attacker(self.args, generate_model=self.generate_model, metrics=self.metrics, device=self.device).get_attacker(epsilon=self.epsilon, global_=True, by_direction=True)
                # 正向攻击
                increase_batch_deltas, batch_true = direction_attacker(batch_all_x, batch_all_y, batch_all_x_mark, batch_all_y_mark, direction='increase')
                # 确保与 batch_all_x 在同设备做相加（均为 CPU）
                increase_batch_deltas = increase_batch_deltas.detach().cpu()
                batch_true = batch_true.detach().cpu()
                increase_pred, _ = self.get_prediction(self.victim_model, batch_all_x + increase_batch_deltas, batch_all_y, batch_all_x_mark, batch_all_y_mark)
                # 逐点计算误差，而不是整体计算
                increase_errors = []
                for i in range(increase_pred.size(0)):
                    # 对每个样本单独计算损失
                    sample_error = calcu_loss(increase_pred[i], batch_true[i])[self.metrics]
                    increase_errors.append(sample_error)
                increase_error = torch.tensor(increase_errors, device=self.device)
                
                # 负向攻击
                decrease_batch_deltas, batch_true = direction_attacker(batch_all_x, batch_all_y, batch_all_x_mark, batch_all_y_mark, direction='decrease')
                decrease_batch_deltas = decrease_batch_deltas.detach().cpu()
                batch_true = batch_true.detach().cpu()
                decrease_pred, _ = self.get_prediction(self.victim_model, batch_all_x + decrease_batch_deltas, batch_all_y, batch_all_x_mark, batch_all_y_mark)
                decrease_errors = []
                for i in range(decrease_pred.size(0)):
                    # 对每个样本单独计算损失
                    sample_error = calcu_loss(decrease_pred[i], batch_true[i])[self.metrics]
                    decrease_errors.append(sample_error)
                decrease_error = torch.tensor(decrease_errors, device=self.device)
                
                # 创建方向标记张量：1表示增加方向更好，0表示减少方向更好
                batch_directions = (increase_error > decrease_error).float()
                all_directions.append(batch_directions)
                
                # 哪个攻击结果好留哪个
                batch_deltas = torch.zeros_like(increase_batch_deltas)
                for i in range(batch_deltas.size(0)):
                    if increase_error[i] > decrease_error[i]:
                        batch_deltas[i] = increase_batch_deltas[i]
                    else:
                        batch_deltas[i] = decrease_batch_deltas[i]

            # print(f"batch_deltas: {batch_deltas.shape}")
            
            # 存储当前批次的数据和delta（累计存储转回 CPU，并断开计算图）
            batch_deltas = batch_deltas.detach().cpu()
            batch_true = batch_true.detach().cpu()

            all_x.append(batch_all_x.cpu())
            all_y.append(batch_all_y.cpu())
            all_x_mark.append(batch_all_x_mark.cpu())
            all_y_mark.append(batch_all_y_mark.cpu())
            all_deltas.append(batch_deltas.cpu())
            all_true.append(batch_true.cpu())

            # 释放批次局部变量引用
            del batch_all_x, batch_all_y, batch_all_x_mark, batch_all_y_mark, batch_deltas, batch_true
            torch.cuda.empty_cache()

        # 合并所有批次的数据
        all_x = torch.cat(all_x, dim=0).float()
        all_y = torch.cat(all_y, dim=0).float()
        all_x_mark = torch.cat(all_x_mark, dim=0).float()
        all_y_mark = torch.cat(all_y_mark, dim=0).float()
        all_deltas = torch.cat(all_deltas, dim=0).float()
        all_true = torch.cat(all_true, dim=0).float()
        
        # 如果是方向攻击，保存原始数据和对应的攻击方向
        if by_direction and len(all_directions) > 0:
            all_directions = torch.cat(all_directions, dim=0).float()
            # 保存原始数据和攻击方向
            # 在文件路径中包含 seq_len 和 pred_len，以适配不同的输入输出长度，并加入 seed
            file_path = f'./classify/{self.args.data}_{self.args.generate_model}_{self.args.seq_len}_{self.args.pred_len}_s{self.args.seed}_attack_directions.pt'
            torch.save({
                'original_data': all_x.cpu(),
                'attack_directions': all_directions.cpu()
            }, file_path)
            print(f"已保存攻击方向数据到 {file_path}")

        return all_x, all_y, all_x_mark, all_y_mark, all_deltas, all_true

    def _get_sample_input(self):
        # 获取样本输入数据，不包含 delta
        batch_x, batch_y, seq_x_mark, seq_y_mark = self.data_set[0]
            
        # 检查并打印数据形状
        print(f"单个样本形状: batch_x: {batch_x.shape}, batch_y: {batch_y.shape}")
        print(f"单个样本形状: seq_x_mark: {seq_x_mark.shape}, seq_y_mark: {seq_y_mark.shape}")
        
        all_x = []
        all_y = []
        all_x_mark = []
        all_y_mark = []
        all_true = []
        
        # 设置批处理大小：统一从 args.batch_size 读取
        batch_size = getattr(self.args, 'batch_size', None)
        if batch_size is None or batch_size <= 0:
            batch_size = 64
        
        # 分批处理数据集
        for i in tqdm(range(0, len(self.data_set), batch_size), desc="获取数据样本"):
            # 获取当前批次的索引范围
            batch_indices = range(i, min(i + batch_size, len(self.data_set)))
            
            # 为当前批次创建张量
            batch_all_x = []
            batch_all_y = []
            batch_all_x_mark = []
            batch_all_y_mark = []
            
            # 收集当前批次的数据（保持在 CPU，计算时再搬到 GPU）
            for j in batch_indices:
                batch_x, batch_y, seq_x_mark, seq_y_mark = self.data_set[j]
                batch_all_x.append(torch.as_tensor(batch_x))
                batch_all_y.append(torch.as_tensor(batch_y))
                batch_all_x_mark.append(torch.as_tensor(seq_x_mark))
                batch_all_y_mark.append(torch.as_tensor(seq_y_mark))
            
            # 将列表转换为张量
            batch_all_x = torch.stack(batch_all_x)
            batch_all_y = torch.stack(batch_all_y)
            batch_all_x_mark = torch.stack(batch_all_x_mark)
            batch_all_y_mark = torch.stack(batch_all_y_mark)
            
            # 从 batch_y 中提取 all_true（预测长度部分）
            batch_true = batch_all_y[:, -self.args.pred_len:, :]
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_true = batch_true[:, :, f_dim:]
            
            # 存储当前批次的数据（累计存储转回 CPU）
            all_x.append(batch_all_x.cpu())
            all_y.append(batch_all_y.cpu())
            all_x_mark.append(batch_all_x_mark.cpu())
            all_y_mark.append(batch_all_y_mark.cpu())
            all_true.append(batch_true.cpu())
            
            # 释放批次局部变量引用
            del batch_all_x, batch_all_y, batch_all_x_mark, batch_all_y_mark, batch_true
            torch.cuda.empty_cache()

        # 合并所有批次的数据
        all_x = torch.cat(all_x, dim=0).float()
        all_y = torch.cat(all_y, dim=0).float()
        all_x_mark = torch.cat(all_x_mark, dim=0).float()
        all_y_mark = torch.cat(all_y_mark, dim=0).float()
        all_true = torch.cat(all_true, dim=0).float()

        return all_x, all_y, all_x_mark, all_y_mark, None, all_true

    def get_prediction(self, model, x, y, seq_x_mark, seq_y_mark, show=True):

        # 对 x,y,seq_x_mark,seq_y_mark 划分 batch 为 self.args.batch_size

        model.eval()

        with torch.no_grad():

            # 划分批次处理数据
            # 预测阶段批大小，优先使用 predict_batch_size，其次 batch_size，最后默认 64
            batch_size = getattr(self.args, 'predict_batch_size', None)
            if batch_size is None or batch_size <= 0:
                batch_size = getattr(self.args, 'batch_size', 64) or 64
            total_samples = x.shape[0]

            # print(f"batch_size: {batch_size}")
            
            # 初始化存储预测结果和真实值的列表
            all_outputs = []
            all_batch_y = []
            
            # 按批次处理数据
            for i in range(0, total_samples, batch_size):
                if show:
                    tqdm.write(f"处理预测批次: {i // batch_size + 1}/{total_samples // batch_size + 1}", end="\r")
                end_idx = min(i + batch_size, total_samples)
                
                # 获取当前批次的数据
                batch_x_i = x[i:end_idx]
                batch_y_i = y[i:end_idx]
                batch_x_mark_i = seq_x_mark[i:end_idx]
                batch_y_mark_i = seq_y_mark[i:end_idx]
                
                # 获取当前批次的预测结果和真实值
                outputs_i, batch_y_i = self._get_prediction_single_batch(model, batch_x_i, batch_y_i, batch_x_mark_i, batch_y_mark_i)
                
                # 存储当前批次的结果（转回 CPU 并断开计算图，以免累计占用显存）
                all_outputs.append(outputs_i.detach().cpu())
                all_batch_y.append(batch_y_i.detach().cpu())
            
            # 合并所有批次的结果
            outputs = torch.cat(all_outputs, dim=0)
            batch_y = torch.cat(all_batch_y, dim=0)
            
            # 返回合并后的结果
            return outputs, batch_y
            
    def _get_prediction_single_batch(self, model, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """处理单个批次的预测"""
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, :].cpu()
        batch_y = batch_y[:, -self.args.pred_len:, :].cpu()
        outputs = outputs[:, :, f_dim:]
        batch_y = batch_y[:, :, f_dim:]

        return outputs, batch_y

    def sample_to_original_data(self, sample):
        '''
        将滑动窗口切割的样本 -> 原始时间序列数据
        '''
        # 计算原始数据长度
        len = sample.shape[0] + self.time_window - 1
        # 将样本转换为原始数据
        original_data = torch.zeros((len, self.feature_num)).cpu()
        # print(f"len: {self.len}")
        # print(f"sample: {sample.shape}")
        for i in range(sample.shape[0]):
            if i == 0:
                original_data[0:self.time_window,:] = sample[0,:,:]
            else:
                original_data[i+self.time_window-1,:] = sample[i,-1,:]
        # 将列表转换为张量
        original_data = original_data.cpu()
        return original_data
    
    def original_data_to_sample(self, original_data):
        '''
        将原始时间序列数据 -> 滑动窗口切割的样本
        '''
        # 将原始数据转换为样本
        sample = []
        for i in range(self.sample_num):
            sample_window = original_data[i:i+self.time_window]
            sample.append(sample_window)
        # 将列表转换为张量
        sample = torch.stack(sample, dim=0)
        sample = sample.cpu()
        return sample
    
    def sample_to_candidate(self, sample):
        '''
        将滑动窗口切割的样本 -> 梯形决策空间
        '''
        # 获取空间
        origin_data_with_candidate = torch.zeros((self.len, self.space, self.feature_num)).cpu()
        
        # 矩阵 -> 梯形决策空间
        dot_num_index = torch.zeros(self.len)
        for i in tqdm(range(self.sample_num), desc="形成决策空间", dynamic_ncols=True):
            for j in range(self.time_window):
                num_index = int(dot_num_index[i + j].item())
                origin_data_with_candidate[i + j, num_index, :] = sample[i, j, :]
                dot_num_index[i + j] = dot_num_index[i + j] + 1

        # 填补梯形前后的角
        for i in range(self.time_window):
            space = i + 1
            for j in range(self.time_window):
                # 从space随机抽一个数
                random_idx = torch.randint(0, space, (1,)).item()
                # 前面的角
                origin_data_with_candidate[i, j, :] = origin_data_with_candidate[i, random_idx, :]
                # 后面的角
                origin_data_with_candidate[self.len - i - 1, j, :] = origin_data_with_candidate[self.len -i - 1, random_idx, :] 
        
        return origin_data_with_candidate
    
    def inverse_transform(self, data):
        # 逆归一化
        shape = data.shape
        data = self.data_set.inverse_transform(data.reshape(shape[0] * shape[1], -1)).reshape(shape)
        return data
    
    def save_sample_delta(self, sample_delta):
        if not self.args.save_sample_delta:
            print(f"不保存样本delta")
            return
        # 保存样本delta
        if not os.path.exists(f"./deltas/{self.args.attack_algo}"):
            os.makedirs(f"./deltas/{self.args.attack_algo}")
        path = f"./deltas/{self.args.attack_algo}/{self.args.data}_{self.args.seq_len}_{self.args.pred_len}_{self.args.generate_model}_{self.args.attack_algo}_{self.args.kind}_e{self.args.epsilon}_at{self.args.alpha_times}_eo{self.args.epoch}_s{self.args.seed}.npy"
        np.save(path, sample_delta.cpu().numpy())

    def save_sample_prediction(self, sample_prediction):
        # 保存样本delta
        if not os.path.exists(f"./predictions/{self.args.attack_algo}"):
            os.makedirs(f"./predictions/{self.args.attack_algo}")
        path = f"./predictions/{self.args.attack_algo}/{self.args.data}_{self.args.seq_len}_{self.args.pred_len}_{self.args.generate_model}_{self.args.attack_algo}_{self.args.kind}_e{self.args.epsilon}_at{self.args.alpha_times}_eo{self.args.epoch}_s{self.args.seed}.npy"
        np.save(path, sample_prediction.cpu().numpy())

    def check_delta_exist(self, kind = None):
        # 判断参数
        kind = self.args.kind if kind is None else kind
        # 检查是否存在 delta
        path = f"./deltas/{self.args.attack_algo}/{self.args.data}_{self.args.seq_len}_{self.args.pred_len}_{self.args.generate_model}_{self.args.attack_algo}_{kind}_e{self.args.epsilon}_at{self.args.alpha_times}_eo{self.args.epoch}_s{self.args.seed}.npy"
        print(f"checking delta exist: {path}")
        if os.path.exists(path):
            return True
        else:
            return False
        
    def load_sample_delta(self, kind = None):
        # 判断参数
        kind = self.args.kind if kind is None else kind
        # 加载样本delta
        path = f"./deltas/{self.args.attack_algo}/{self.args.data}_{self.args.seq_len}_{self.args.pred_len}_{self.args.generate_model}_{self.args.attack_algo}_{kind}_e{self.args.epsilon}_at{self.args.alpha_times}_eo{self.args.epoch}_s{self.args.seed}.npy"
        sample_delta = np.load(path)
        # sample_delta = np.load(f"./deltas/{self.args.data}_{self.args.seq_len}_{self.args.pred_len}_{self.args.generate_model}_{kind}_{self.args.attack_algo}_e{self.args.epsilon}.npy")

        # 返回 CPU 张量，避免与 all_x CPU 相加时设备不一致
        return torch.from_numpy(sample_delta).cpu()
    
    def get_relevant_indices(self, index):
        '''
        获取时间序列中 index 所在位置的相关滑动窗口样本下标
        '''
        indicator = torch.zeros(self.len, self.feature_num)
        indicator[index, :] = 1
        sample_indicator = self.original_data_to_sample(indicator)

        # 寻找相关样本,即 indicator 存在 1 的样本
        relevant_indices = []
        for j in range(self.sample_num):
            if torch.sum(sample_indicator[j]) > 0:
                relevant_indices.append(j)
        return relevant_indices
    
    def get_candidate_delta(self, all_candidate, best_choice):
        # 获取每个时点最佳的 delta
        all_candidate_delta = torch.zeros_like(all_candidate[:,0,:])
        for i in range(all_candidate.size()[0]):
            all_candidate_delta[i,:] = all_candidate[i,int(best_choice[i]),:]
        return all_candidate_delta
    
    def cal_loss(self, all_prediction, all_true, final = True):
        
        loss = calcu_loss(all_prediction, all_true)

        if final:
            print("----------------------------------------")
            print(f"攻击后的损失值: {loss[self.metrics]:.4f}")

            # 更新记录
            record_name = getattr(self.args, "record_name", "record")
            record_filename = f'./{record_name}.csv'
            update_record(
                args=self.args,
                matrics=self.metrics,
                loss=loss,
                solution=None,
                filename=record_filename
            )

        return loss[self.metrics]
    
    def convert_to_global(self, all_deltas, all_x, all_y, all_x_mark, all_y_mark, all_true):

        # raw
        self.kind = 'raw'
        self.args.kind = 'raw'
        self.save_sample_delta(all_deltas)
        # 获取预测值
        all_prediction, _ = self.get_prediction(self.victim_model, all_x + all_deltas.cpu(), all_y, all_x_mark, all_y_mark)
        # 保存样本预测值
        if self.args.save_prediction:
            self.save_sample_prediction(all_prediction)
        # 计算损失
        self.cal_loss(all_prediction, all_true)

        all_candidate = self.sample_to_candidate(all_deltas)

        # first 
        self.kind = 'first'
        self.args.kind = 'first'
        selected_deltas = all_candidate[:,0,:]
        # 将 delta 转换为样本形式
        all_deltas = self.original_data_to_sample(selected_deltas)
        # 保存样本delta
        self.save_sample_delta(all_deltas)
        # 获取预测值
        all_prediction, _ = self.get_prediction(self.victim_model, all_x + all_deltas.cpu(), all_y, all_x_mark, all_y_mark)
        # 保存样本预测值
        if self.args.save_prediction:
            self.save_sample_prediction(all_prediction)
        # 计算损失
        self.cal_loss(all_prediction, all_true)

        # last
        self.kind = 'last'
        self.args.kind = 'last'
        selected_deltas = all_candidate[:,self.space-1,:]
        # 将 delta 转换为样本形式
        all_deltas = self.original_data_to_sample(selected_deltas)
        # 保存样本delta
        self.save_sample_delta(all_deltas)
        # 获取预测值
        all_prediction, _ = self.get_prediction(self.victim_model, all_x + all_deltas.cpu(), all_y, all_x_mark, all_y_mark)
        # 保存样本预测值
        if self.args.save_prediction:
            self.save_sample_prediction(all_prediction)
        # 计算损失
        self.cal_loss(all_prediction, all_true)

        # random
        self.kind = 'random'
        self.args.kind = 'random'
        for i in range(all_candidate.size()[0]):
            random_index = torch.randint(0, self.space, (1,)).item()
            selected_deltas[i,:] = all_candidate[i, random_index, :]
        # 将 delta 转换为样本形式
        all_deltas = self.original_data_to_sample(selected_deltas)
        # 保存样本delta
        self.save_sample_delta(all_deltas)
        # 获取预测值
        all_prediction, _ = self.get_prediction(self.victim_model, all_x + all_deltas.cpu(), all_y, all_x_mark, all_y_mark)
        # 保存样本预测值
        if self.args.save_prediction:
            self.save_sample_prediction(all_prediction)
        # 计算损失
        self.cal_loss(all_prediction, all_true)
        
        