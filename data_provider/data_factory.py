from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'ILI': Dataset_Custom,
    'Weather': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    """数据提供函数, 旨在根据实验配置和任务类型动态加载并返回 是数据加载流程的核心，提供了适配多任务、多数据集的灵活接口

    Args:
        args (dic): 存放着用于构建dataset和dataloader的参数
        flag (str): 'train', 'val' or 'test' 用于标签取数据集中哪个部分的样本

    Returns:
        dataset, dataloader: 返回flag对应的数据集中的样本及其迭代器 data_set: 数据集对象, 封装了原始数据, 并提供了切片和预处理等方法 data_loader: 数据加载器对象, 将 data_set中的样本组织成批次并提供迭代功能
    
    """
    """
    说明:
        data_set 是数据集对象，封装了原始数据，并定义了如何获取单个样本
            管理和访问数据: 存储原始数据并提供访问逻辑
            切片和预处理: 定义数据的切片方式（如时间序列的窗口化）以及预处理操作（如归一化、标准化）
            实现核心方法: 
                __len__: 返回数据集的样本数量
                __getitem__: 根据索引返回一个样本
        data_loader 是数据加载器，用于将 data_set 中的样本组织成批次并提供迭代功能。
            批量化: 将数据分割成批次(batch), 每次返回一个批量数据
            多线程加载: 通过 num_workers 提高数据加载效率
            打乱顺序: 通过 shuffle=True 打乱数据顺序
            自动迭代: 支持 for 循环或手动迭代访问
    """
    Data = data_dict[args.data]  # 加载相应数据集处理类
    timeenc = 0 if args.embed != 'timeF' else 1 # 通过传入的embed参数选择时间编码模式
    # 数据打乱标志
    # 测试集保持顺序不变，以确保结果可复现;训练集打乱数据，有助于提升模型训练的泛化能力。
    # 不同的批次可能数据加载顺序随机化
    shuffle_flag = False
    drop_last = False  # 丢弃最后一个不足 batch_size 的批次
    batch_size = args.batch_size
    freq = args.freq

    # 获得数据集
    data_set = Data(
        args = args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )
    print(flag, len(data_set))  # 输出数据用途和数据集样本数量
    # 封装数据集，便于批量加载
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


if __name__ == '__main__':
    import argparse

    args = argparse.Namespace()
    args.data = 'PVOD'
    args.batch_size = 32
    args.freq = '15min'
    args.augmentation_ratio = 0
    args.flag = 'train'


    Data = data_dict[args.data]
    data_set = Data(
        args = args,
        flag=args.flag
    )
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size
    )
    for x, y, x_mark, y_mark in data_loader:
        print(x.shape)
        print(y.shape)
        print(x_mark.shape)
        print(y_mark.shape)
        break
    