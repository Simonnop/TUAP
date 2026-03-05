# 建立csv: generate_model,victim_model,data,epsilon,attack_rate,action_type,mask_type,matrices_best,seed_best,matrices_worst,seed_worst
# 每一次 attack 执行后, 读取 csv, 找到参数对应的行, 比较最好与最差的 matrices, 如果更好或者更差, 就替换该值和对应的seed, 存储对应的 solution 为 npy 文件,用参数命名

import os
import pandas as pd
import numpy as np

def init_record_file(filename):
    """初始化记录文件"""
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=[
            'generate_model', 'victim_model', 'data', 'epsilon',
            'attack_rate', 'epoch', 'alpha_times', 'mu', 'kind', 'attack_algo',
            'seq_len', 'pred_len', 'seed'
        ])
        df.to_csv(filename, index=False)

def get_solution_filename(args):
    """生成solution文件名"""
    mu = getattr(args, 'mu', 1.0)
    filename = f"solutions/{args.kind}/{args.data}/{args.generate_model}_{args.victim_model}/"\
           f"eps{args.epsilon}_rate{args.attack_rate}_epoch{args.epoch}_alpha{args.alpha_times}_mu{mu}_"\
           f"seq{args.seq_len}_pred{args.pred_len}_{args.attack_algo}_seed{args.seed}"
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    return filename

def update_record(args, loss: dict, matrics = 'mse', solution = None, filename='./record.csv'):
    """更新记录
    
    Args:
        args: 参数对象,包含generate_model,victim_model等
        matrices: 当前matrices值
        seed: 当前seed值
        solution: 当前solution数组
        filename: 记录文件路径
    """

    # 
    init_record_file(filename)

    df = pd.read_csv(filename)
    # 兼容旧版 csv 无 mu 列
    if 'mu' not in df.columns:
        df['mu'] = 1.0
    
    # 构建查询条件
    params = {
        'generate_model': args.generate_model,
        'victim_model': args.victim_model,
        'data': args.data,
        'epsilon': args.epsilon,
        'attack_rate': args.attack_rate,
        'attack_algo': args.attack_algo,
        'kind': args.kind + '_' + args.sort_by if args.kind == 'grad_greedy' else args.kind,
        'epoch': args.epoch,
        'alpha_times': args.alpha_times,
        'mu': getattr(args, 'mu', 1.0),
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'seed': args.seed
    }
    
    query = ' & '.join([f"{k}=='{v}'" if isinstance(v, str) else f"{k}=={v}" 
                       for k,v in params.items()])
    
    row = df.query(query)
    
    if len(row) == 0:
        # 新建记录
        new_record = params.copy()
        new_record.update({k: round(v, 5) for k, v in loss.items()})
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        
        # 保存当前solution
        if solution is not None:
            np.save(f"{get_solution_filename(args)}.npy", solution)
        
    else:
        idx = row.index[0]
        if (args.kind != 'worse' and loss[matrics] > df.loc[idx, matrics]) or (args.kind == 'worse' and loss[matrics] < df.loc[idx, matrics]):
            # 更新最佳记录
            loss = {k: round(v, 5) for k, v in loss.items()}

            for key in loss.keys():
                df.loc[idx, key] = loss[key]
            if solution is not None:
                np.save(f"{get_solution_filename(args)}.npy", solution)
    
    df.to_csv(filename, index=False)

def load_solution(args):
    """
    加载指定参数下的最优或最差解决方案
    
    Args:
        args: 参数配置
        type: 'best' 或 'worst', 指定加载最优还是最差解决方案
        
    Returns:
        numpy.ndarray: 加载的解决方案数组
    """
    filename = f"{get_solution_filename(args)}.npy"
    return np.load(filename)
