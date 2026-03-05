
# Considering SVM classifier works well on small sample sets, nonlinear problems and high-dimensional patterns [53], it is adopted to train the two classification models.

# 使用 SVM 分类器, 训练攻击方向分类器

# Moreover, for the purpose of avoiding the performance degradation caused by the sample imbalance problem, Borderline SMOTE [54] data synthesis algorithm is used to oversample the samples.

# 使用 Borderline SMOTE 数据合成算法, 解决数据的不平衡问题

# Therefore, in this paper, p is set at 20% and 30%, and epsilon is set to be 10%, 15%, and 20%.

# 设置 p 为 20% 和 30%, 设置 epsilon 为 10%, 15%, 和 20%.

# In order to evaluate the actual attack effect improved by the ADJM, nine forecasting models are attacked by random direction attack (RDattack), judging direction attack (JDattack) and direction known attack (DKattack)

# 为了评估 ADJM 实际的攻击效果, 使用随机方向攻击 (RDattack), 判断方向攻击 (JDattack) 和方向已知攻击 (DKattack)

import os
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from exp.attack.exp_attack_basic import Exp_Attack_Basic
from exp.utils.attacker import Attacker

class Exp_Adjm_Method(Exp_Attack_Basic):

    def __init__(self, args):
        super(Exp_Adjm_Method, self).__init__(args)
        # SVM分类器
        self.svm_model = None
        self.scaler = None

    def attack(self):

        self.svm_model, self.scaler = self._get_classify_model()

        direction_attacker = Attacker(self.args, generate_model=self.generate_model, metrics=self.metrics, device=self.device).get_attacker(epsilon=self.epsilon, global_=True, by_direction=True)

        # 给攻击的数据预测方向
        # 获取样本和 delta
        all_x, all_y, all_x_mark, all_y_mark, all_deltas, all_true = self._get_sample_input()
        predictions = self.predict_attack_direction(all_x)

        all_deltas = torch.zeros_like(all_x)

        # 根据方向进行攻击
        from tqdm import tqdm
        for i in tqdm(range(all_x.size(0)), desc="根据预测方向进行攻击"):
            delta, true = direction_attacker(all_x[i], all_y[i], all_x_mark[i], all_y_mark[i], predictions[i])
            # 将结果移回 CPU 以保持与 all_x 相同的设备
            all_deltas[i] = delta.cpu()
            all_true[i] = true.cpu()

        # 获取预测值
        # 确保 all_deltas 在 CPU 上以便与 all_x 相加
        all_deltas = all_deltas.cpu()
        self.convert_to_global(all_deltas, all_x, all_y, all_x_mark, all_y_mark, all_true)
    
    def predict_attack_direction(self, data, load_model=False):
        """
        使用训练好的SVM模型预测攻击方向
        
        参数:
        data: 需要预测的数据，形状为 [batch_size, seq_len, feature_dim]
        load_model: 是否加载已保存的模型，默认为False
        
        返回:
        预测的攻击方向，形状为 [batch_size]，值为0或1
        """
        
        # 将PyTorch张量转换为NumPy数组
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        # 重塑数据为2D形式 [samples, features]
        # 适配不同的输入输出长度：支持 [batch_size, seq_len, feature_dim] 或 [batch_size, feature_dim]
        if len(data.shape) == 3:
            batch_size, seq_len, feature_dim = data.shape
            X = data.reshape(batch_size, -1)  # 将时间序列展平
        elif len(data.shape) == 2:
            batch_size, feature_dim = data.shape
            X = data
        else:
            raise ValueError(f"不支持的数据形状: {data.shape}")
        
        # 数据标准化
        X_scaled = self.scaler.transform(X)
        
        # 预测攻击方向
        predictions = self.svm_model.predict(X_scaled)
        
        # 输出预测结果
        print(f"预测的攻击方向分布: 减少方向: {np.sum(predictions == 0)}, 增加方向: {np.sum(predictions == 1)}")
        
        return predictions

    def _get_classify_model(self):
        # 如果存在SVM分类模型，则加载，否则生成新的SVM分类模型
        # 尝试加载带 seed 的模型，如果不存在则尝试加载不带 seed 的模型
        import joblib
        model_path_with_seed = f'./classify/models/{self.args.data}_{self.args.model}_{self.args.seq_len}_{self.args.pred_len}_s{self.args.seed}_svm_classifier.joblib'
        scaler_path_with_seed = f'./classify/models/{self.args.data}_{self.args.model}_{self.args.seq_len}_{self.args.pred_len}_s{self.args.seed}_scaler.joblib'
        
        model_path_no_seed = f'./classify/models/{self.args.data}_{self.args.model}_{self.args.seq_len}_{self.args.pred_len}_svm_classifier.joblib'
        scaler_path_no_seed = f'./classify/models/{self.args.data}_{self.args.model}_{self.args.seq_len}_{self.args.pred_len}_scaler.joblib'
            
        if os.path.exists(model_path_with_seed) and os.path.exists(scaler_path_with_seed):
            model_path, scaler_path = model_path_with_seed, scaler_path_with_seed
        elif os.path.exists(model_path_no_seed) and os.path.exists(scaler_path_no_seed):
            model_path, scaler_path = model_path_no_seed, scaler_path_no_seed
        else:
            raise FileNotFoundError(f"未找到模型文件 (尝试了带种子和不带种子的路径)")

        print(f"加载模型文件: {model_path}")
        print(f"加载数据标准化器: {scaler_path}")

        svm_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return svm_model, scaler

