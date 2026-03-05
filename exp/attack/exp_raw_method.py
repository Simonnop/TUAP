
from exp.attack.exp_attack_basic import Exp_Attack_Basic

class Exp_Raw_Method(Exp_Attack_Basic):

    def __init__(self, args):
        super(Exp_Raw_Method, self).__init__(args)

    def attack(self):
        
        # 获取样本和delta
        all_x, all_y, all_x_mark, all_y_mark, all_deltas, all_true = self._get_sample_delta()

        self.convert_to_global(all_deltas, all_x, all_y, all_x_mark, all_y_mark, all_true)