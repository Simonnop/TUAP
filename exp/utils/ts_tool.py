import torch

class TimeSeriesTool:
    def __init__(self, sample_num: int, time_window: int, var_num: int):
        self.sample_num = sample_num
        self.time_window = time_window
        self.var_num = var_num
        self.length = sample_num + time_window - 1
        self.position_dict = self._create_position_dict()

    def series_to_sample(self, series: torch.Tensor):
        # 滑动窗口切割
        samples = []
        for i in range(self.sample_num):
            sample = series[i:i+self.time_window,:]
            samples.append(sample)
        return torch.stack(samples)
    
    def _create_position_dict(self):
        # 存储代表相同位置的点
        # 是从 0 开始的顺序连续 dict, 所以用 list
        position_dict = []
        # 两个 index 的和相同为同一个点,用于替换
        for l in range(self.length):
            same_position_list = []
            # 找出同一个点的位置
            for i in range(self.sample_num):
                j = l - i
                if j > self.time_window - 1:
                    continue
                if j < 0:
                    break
                same_position_list.append((i,j))
                # # MARK 调试用, 跳过多点选择
                # break
            position_dict.append(same_position_list)
        return position_dict