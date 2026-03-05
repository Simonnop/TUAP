
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

class Exp_Classify_Direction(Exp_Attack_Basic):

    def __init__(self, args):
        args.flag = 'train'
        super(Exp_Classify_Direction, self).__init__(args)
        # SVM分类器
        self.svm_model = None
        self.scaler = None

    def train(self, setting):
        original_data, attack_directions = self._get_classify_data()
        self.train_svm_classifier(original_data, attack_directions)

    def test(self, setting, test=0):
        original_data, attack_directions = self._get_classify_data()
        self.svm_model, self.scaler = self._get_classify_model()
        self.test_svm_classifier(original_data, attack_directions)

    def _get_classify_data(self):
        # 如果存在攻击方向数据，则加载，否则生成新的攻击方向数据
        # 在文件路径中包含 seq_len 和 pred_len，以适配不同的输入输出长度，并加入 seed
        file_path_with_seed = f'./classify/{self.args.data}_{self.args.generate_model}_{self.args.seq_len}_{self.args.pred_len}_s{self.args.seed}_attack_directions.pt'
        file_path_no_seed = f'./classify/{self.args.data}_{self.args.generate_model}_{self.args.seq_len}_{self.args.pred_len}_attack_directions.pt'
        
        if os.path.exists(file_path_with_seed):
            file_path = file_path_with_seed
        elif os.path.exists(file_path_no_seed):
            file_path = file_path_no_seed
        else:
            print(f"未找到攻击方向数据，生成新的攻击方向数据...")
            self._get_sample_delta(by_direction=True)
            # _get_sample_delta 内部会保存到带 seed 的路径（如果是新代码的话）
            # 我们先假设它保存了，重新检查
            if os.path.exists(file_path_with_seed):
                file_path = file_path_with_seed
            else:
                file_path = file_path_no_seed
                
        print(f"加载攻击方向数据: {file_path}")
        attack_data = torch.load(file_path)
        original_data = attack_data['original_data']
        attack_directions = attack_data['attack_directions']
        return original_data, attack_directions

    def _get_classify_model(self):
        # 如果存在SVM分类模型，则加载，否则生成新的SVM分类模型
        # 在文件路径中包含 seq_len 和 pred_len，以适配不同的输入输出长度
        import joblib
        model_path = f'./classify/models/{self.args.data}_{self.args.generate_model}_{self.args.seq_len}_{self.args.pred_len}_s{self.args.seed}_svm_classifier.joblib'
        scaler_path = f'./classify/models/{self.args.data}_{self.args.generate_model}_{self.args.seq_len}_{self.args.pred_len}_s{self.args.seed}_scaler.joblib'
            
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"模型文件不存在，请先训练模型: {model_path}")
            
        svm_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return svm_model, scaler

    def train_svm_classifier(self, original_data, attack_directions, save_model=True):
        """
        使用SVM分类器训练模型，预测攻击方向
        
        参数:
        original_data: 原始时间序列数据，形状为 [batch_size, seq_len, feature_dim]
        attack_directions: 攻击方向标签，形状为 [batch_size]，值为0或1
        save_model: 是否保存模型，默认为True
        
        返回:
        训练好的SVM模型
        """
        print("开始训练SVM分类器...")
        
        # 将PyTorch张量转换为NumPy数组
        if isinstance(original_data, torch.Tensor):
            original_data = original_data.cpu().numpy()
        if isinstance(attack_directions, torch.Tensor):
            attack_directions = attack_directions.cpu().numpy()
        
        # 重塑数据为2D形式 [samples, features]
        # 适配不同的输入输出长度：支持 [batch_size, seq_len, feature_dim] 或 [batch_size, feature_dim]
        if len(original_data.shape) == 3:
            batch_size, seq_len, feature_dim = original_data.shape
            X = original_data.reshape(batch_size, -1)  # 将时间序列展平
        elif len(original_data.shape) == 2:
            batch_size, feature_dim = original_data.shape
            X = original_data
        else:
            raise ValueError(f"不支持的数据形状: {original_data.shape}")
        y = attack_directions
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 数据标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 创建SVM分类器
        from sklearn.svm import LinearSVC
        # self.svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, verbose=True, cache_size=10000)
        self.svm_model = LinearSVC(C=1.0, verbose=1, max_iter=100)
        
        # 训练模型
        self.svm_model.fit(X_train_scaled, y_train)
        
        # 保存模型
        if save_model:
            import joblib
            model_dir = f'./classify/models'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            # 在文件路径中包含 seq_len 和 pred_len，并加入 seed
            model_path = f'{model_dir}/{self.args.data}_{self.args.generate_model}_{self.args.seq_len}_{self.args.pred_len}_s{self.args.seed}_svm_classifier.joblib'
            scaler_path = f'{model_dir}/{self.args.data}_{self.args.generate_model}_{self.args.seq_len}_{self.args.pred_len}_s{self.args.seed}_scaler.joblib'
            joblib.dump(self.svm_model, model_path)
            joblib.dump(self.scaler, scaler_path)
            print(f"SVM分类器已保存到 {model_path}")
            print(f"数据标准化器已保存到 {scaler_path}")
        
        return self.svm_model

    def test_svm_classifier(self, original_data, attack_directions):
        """
        使用SVM分类器测试模型，预测攻击方向
        
        参数:
        original_data: 原始时间序列数据，形状为 [batch_size, seq_len, feature_dim]
        attack_directions: 攻击方向标签，形状为 [batch_size]，值为0或1
        
        返回:
        训练好的SVM模型
        """
        print("开始测试SVM分类器...")
        
        # 将PyTorch张量转换为NumPy数组
        if isinstance(original_data, torch.Tensor):
            original_data = original_data.cpu().numpy()
        if isinstance(attack_directions, torch.Tensor):
            attack_directions = attack_directions.cpu().numpy()
        
        # 重塑数据为2D形式 [samples, features]
        # 适配不同的输入输出长度：支持 [batch_size, seq_len, feature_dim] 或 [batch_size, feature_dim]
        if len(original_data.shape) == 3:
            batch_size, seq_len, feature_dim = original_data.shape
            X = original_data.reshape(batch_size, -1)  # 将时间序列展平
        elif len(original_data.shape) == 2:
            batch_size, feature_dim = original_data.shape
            X = original_data
        else:
            raise ValueError(f"不支持的数据形状: {original_data.shape}")
        y = attack_directions
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 数据标准化
        X_test_scaled = self.scaler.transform(X_test)
        
        # 评估模型
        y_pred = self.svm_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"SVM分类器准确率: {accuracy:.4f}")
        print("分类报告:")
        print(classification_report(y_test, y_pred, target_names=['减少方向', '增加方向']))