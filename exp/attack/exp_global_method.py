
from exp.attack.exp_attack_basic import Exp_Attack_Basic
from exp.utils.attacker import Attacker

class Exp_Global_Method(Exp_Attack_Basic):

    def __init__(self, args):
        super(Exp_Global_Method, self).__init__(args)

    def attack(self):

        # 获取样本和 delta
        all_x, all_y, all_x_mark, all_y_mark, all_deltas, all_true = self._get_sample_input()

        # 获取原始序列
        original_x = self.sample_to_original_data(all_x)
            
        # 获取攻击器
        attacker = Attacker(self.args, generate_model=self.generate_model, metrics=self.metrics, device=self.device).get_attacker(epsilon=self.epsilon, global_=True)

        # 攻击（all_* 保持在 CPU；attacker 内部会统一搬到 device）
        origin_deltas = attacker(original_x.cpu(), all_x.cpu(), all_y.cpu(), all_x_mark.cpu(), all_y_mark.cpu()).cpu()
        # 获取扰动
        all_deltas = self.original_data_to_sample(origin_deltas)

        # 保存样本delta
        self.save_sample_delta(all_deltas)

        # 获取预测值（在 CPU 上进行相加，内部预测函数会自己处理上/下设备）
        all_prediction, all_true = self.get_prediction(self.victim_model, all_x.cpu() + all_deltas.cpu(), all_y.cpu(), all_x_mark.cpu(), all_y_mark.cpu())

        # 保存样本预测值
        if self.args.save_prediction:
            self.save_sample_prediction(all_prediction)

        # 计算损失
        self.cal_loss(all_prediction, all_true)
