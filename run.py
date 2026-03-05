import argparse
from ast import arg
import os
import torch
import torch.backends
from exp.train.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.train.exp_classify_direction import Exp_Classify_Direction

# from exp.attack.exp_wrong import Exp_Wrong
from exp.attack.exp_raw_method import Exp_Raw_Method
from exp.attack.exp_global_method import Exp_Global_Method
from exp.attack.exp_bo_method import Exp_Bo_Method
from exp.attack.exp_adjm_method import Exp_Adjm_Method

from utils.print_args import print_args  # 新增参数过多 待为新增参数添加打印形式
import random
import numpy as np
from threadpoolctl import threadpool_limits

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # argparse.ArgumentParser: 用于创建一个命令行参数解析器,接受用户输入的运行配置 参数description为程序的描述信息
    parser = argparse.ArgumentParser(description='Time-Series-Library')


    parser.add_argument(
        "--kind",
        type=str,
        default="optimize_both",
        required=False,
        help="攻击类型, options: [optimize_delta]",
    )

    parser.add_argument(
        "--attack_algo",
        type=str,
        default="FGSM",
        required=False,
        help="攻击算法, options: [FGSM, MIFGSM, PGD]"
    )

    parser.add_argument(
        "--generate_model",
        type=str,
        required=False,
        default="none",
        help="model name, options: [...]",
    )

    parser.add_argument(
        "--victim_model",
        type=str,
        required=False,
        default="ANN",
        help="model name, options: [...]",
    )

    parser.add_argument("--epsilon", type=float, default=0.1, help="攻击的扰动大小")
    parser.add_argument("--epoch", type=int, default=10, help="攻击的迭代次数")
    parser.add_argument("--alpha_times", type=float, default=1, help="攻击的步长倍数 即 alpha = epsilon / epoch * alpha_times")
    parser.add_argument("--mu", type=float, default=1.0, help="MI系列攻击的动量衰减因子 (decay)")
    parser.add_argument("--attack_rate", type=float, default=1, help="攻击的比例")

    # 对抗防御相关参数
    parser.add_argument("--defense_method", type=str, default="pgd_at",
                        help="对抗防御方法, options: [pgd_at, trades]")
    parser.add_argument("--adv_epsilon", type=float, default=0.1, help="对抗训练扰动大小")
    parser.add_argument("--adv_steps", type=int, default=3, help="对抗训练迭代步数")
    parser.add_argument("--adv_alpha", type=float, default=None, help="对抗训练步长, 默认 epsilon/steps")
    parser.add_argument("--adv_norm", type=str, default="linfty", help="对抗扰动范数, options: [linfty, l2]")
    parser.add_argument("--adv_beta", type=float, default=1.0, help="TRADES 正则项权重")
    
    # 是否保存 prediction
    parser.add_argument("--save_prediction", type=int, default=0, help="是否保存预测结果")
    parser.add_argument("--save_sample_delta", type=int, default=0, help="是否保存样本 delta")

    # ⬆️ 攻击任务专用参数
    # ---------------------------------------------------------------
    # ⬇️ 训练相关参数

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='用于判断调用什么实验 在本页面查找"match args.task_name"以查看支持的实验')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='',
                        help='model_id 单纯用于命名结果储存的文件夹、寻找对应的最优超参数组合、参数文件。为空则以task_name_model_data为默认值')
    parser.add_argument('--model', type=str, required=False, default='Autoformer', help='见exp_basic中的model_dict和attacker_dict 在攻击任务中指被攻击模型')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--logs', type=str, default='./logs/', help='日志文件记录位置')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')  # 指定时间序列的季节性模式，适用于像 M4 数据集这种包含不同时间序列特性的任务，决定当前运行任务的数据子集
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, 
                        help='指定缺失数据的掩码比率 用户可以控制模型在训练过程中要"遮掩"多少比例的输入数据 从而进行缺失数据的预测和插补 设置该参数后 模型会在训练时随机选择一定比例的数据点进行遮掩 从而迫使模型学习如何根据已知数据预测和恢复缺失部分')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')  # 写单独的百分号会报错

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='模型隐藏层维度')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='全连接层隐藏层维度')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='注意力机制的缩放因子')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF基于时间特征的编码, fixed固定编码, learned可学习权重进行编码]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='加载数据或使用mealpy求解问题时使用cpu的个数')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--global_chunk_size', type=int, default=64, help='global attack forward chunk size to limit GPU memory')
    parser.add_argument('--predict_batch_size', type=int, default=128, help='prediction batch size to limit GPU memory')

    # GPU
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用的gpu设备序号。注: 使用单GPU时设置 会将设备设为环境变量; 使用多GPU时 只需配置use_multi_gpu和devices')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus.注: use_multi_gpu为真时才有效 会将设备设为环境变量')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')  # nargs: ? 0 或 1 个值  * 任意数量值（包括 0 个） + 至少 1 个值  int 必须精确提供指定数量的值
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    # DTW 是一种计算两个时间序列之间相似性的方法,允许时间序列在时间轴上进行非线性拉伸或收缩，从而找到最佳匹配路径
    # 适用于动态变化的时间序列（如传感器数据、时序信号）
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation数据增强
    '''
        Augmentation模块 是一种数据预处理技术
        主要用于通过对已有数据进行变换和扩展，生成更多样化的数据
        提升模型的泛化能力，减少过拟合问题
        这个模块通常用于深度学习和机器学习中的数据预处理阶段
    '''
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # Tune_forecast
    parser.add_argument('--cpu_per_trial', type=float, default=1, help="使用ray框架调超参数时 需指定的每次实验占用的cpu资源")
    parser.add_argument('--gpu_per_trial', type=float, default=0.5, help="使用ray框架调超参数时 需指定的每次实验占用的gpu资源")
    parser.add_argument('--optim', type=str, default='Adam', help="优化器 定义于exp/exp_basic.py")

    # Attack
    # parser.add_argument('--attacker', type=str, default='mifgsm', help='攻击算法 定义于exp/exp_basic.py')
    # parser.add_argument('--surrogate', type=str, default='', help='代理模型 用于exp/exp_basic.py测试实验定义读取对抗样本的位置 默认将与attacker保持一致')
    # parser.add_argument('--epsilon', type=float, default=16/255, help='阈值')
    # parser.add_argument('--alpha', type=float, default=1.6/255, help='the stepsize to update the perturbation')
    # parser.add_argument('--momentum', type=float, default=0., help='the decay factor for momentum based attack')
    # parser.add_argument('--random_start', action='store_true', default=False, help='set random start')
    # parser.add_argument('--norm', type=str, default='linfty', help='正则化方法')
    # parser.add_argument('--decay', type=float, default=1.0, help='衰减')

    # Tune_attack
    parser.add_argument('--solve_mode', type=str, default='single', help='使用mealpy求解问题时所使用的模式')

    # Graph
    parser.add_argument('--graph_flag', default=None,
                        help='为了将图纳入框架内 模型输入的形状改变在模型内完成 可通过该参数设定形状。注: 该参数不需要手动配置(攻击的测试模式下需传入 因为此时不读取测试集) 在构造训练集时载入。')
    parser.add_argument('--graph_info', default=None,
                        help='一般指邻接矩阵。注：该参数不需要手动配置 在构造训练集时载入。')
    
    # save config
    parser.add_argument('--save_dict_only', action='store_true', default=False, help='保存模型时 是否只保存模型的参数字典；推荐关闭 否则需要在攻击时也要传入模型超参数')
    parser.add_argument('--record_name', type=str, default='record', help='指定该任务下输出的 csv 文件的文件名（不含扩展名，扩展名固定为.csv）')

    args = parser.parse_args()  # 实例化
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')
    
    if args.generate_model != 'none':
        args.model = args.generate_model

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    match args.attack_algo:
        case 'FGSM':
            args.kind = 'raw'
        case 'MIFGSM':
            args.kind = 'raw'
        case 'PGD':
            args.kind = 'raw'
        case 'GGAA':
            args.kind = 'global'
        case 'TCA':
            args.kind = 'raw'
        case 'BIM':
            args.kind = 'raw'
        case 'ATSG':
            args.kind = 'raw'
        case 'BO':
            args.kind = 'raw'
        case 'ADJM':
            args.kind = 'raw'
        case 'NIFGSM':
            args.kind = 'raw'
        case 'VMIFGSM':
            args.kind = 'raw'
        case 'GGAA_NIFGSM':
            args.kind = 'global'
        case 'GGAA_VMIFGSM':
            args.kind = 'global'
        case 'GGAA_FGSM':
            args.kind = 'global'
        case 'GGAA_BIM':
            args.kind = 'global'
        case 'GGAA_PGD':
            args.kind = 'global'
        case 'IEFGSM' | 'PIFGSM' | 'GIFGSM':
            args.kind = 'raw'
        case 'GGAA_IEFGSM':
            args.kind = 'global'
        case 'GTW_Fix':
            args.kind = 'global'
        case 'GGAA_First' | 'GGAA_Last' | 'GGAA_Random':
            args.kind = 'global'
        case 'GGAA_GIFGSM' | 'GGAA_PIFGSM':
            args.kind = 'global'
        case _:
            raise ValueError('未能找到支持的攻击算法')

    # print('Args in experiment:')
    # print_args(args)
    print(args.task_name)
    print(args.kind)
    if args.task_name == 'attack':
        if 'aug' in args.kind:
            raise ValueError('aug 任务未实现')
        
        for ii in range(args.itr):
            # if ii != 0:
            #     continue
            # 设置随机种子
            current_seed = ii
            set_seed(current_seed)
            args.seed = current_seed
            print(f'>>>>>>>start attack round {ii} with seed {current_seed}>>>>>>>>>>>>>>>>>>>>>>>>>>')

            match args.attack_algo:
                case 'FGSM':
                    Exp = Exp_Raw_Method(args)
                case 'MIFGSM':
                    Exp = Exp_Raw_Method(args)
                case 'PGD':
                    Exp = Exp_Raw_Method(args)
                case 'TCA':
                    Exp = Exp_Raw_Method(args)
                case 'BIM':
                    Exp = Exp_Raw_Method(args)
                case 'ATSG':
                    Exp = Exp_Raw_Method(args)
                case 'BO':
                    Exp = Exp_Bo_Method(args)
                case 'ADJM':
                    Exp = Exp_Adjm_Method(args)
                case 'GGAA':
                    Exp = Exp_Global_Method(args)
                case 'NIFGSM':
                    Exp = Exp_Raw_Method(args)
                case 'VMIFGSM':
                    Exp = Exp_Raw_Method(args)
                case 'GGAA_NIFGSM':
                    Exp = Exp_Global_Method(args)
                case 'GGAA_VMIFGSM':
                    Exp = Exp_Global_Method(args)
                case 'GGAA_FGSM':
                    Exp = Exp_Global_Method(args)
                case 'GGAA_BIM':
                    Exp = Exp_Global_Method(args)
                case 'GGAA_PGD':
                    Exp = Exp_Global_Method(args)
                case 'IEFGSM' | 'PIFGSM' | 'GIFGSM':
                    Exp = Exp_Raw_Method(args)
                case 'GGAA_IEFGSM':
                    Exp = Exp_Global_Method(args)
                case 'GTW_Fix':
                    Exp = Exp_Global_Method(args)
                case 'GGAA_First' | 'GGAA_Last' | 'GGAA_Random':
                    Exp = Exp_Global_Method(args)
                case 'GGAA_GIFGSM' | 'GGAA_PIFGSM':
                    Exp = Exp_Global_Method(args)
            
            with threadpool_limits(limits=5, user_api="blas"):
                print("数据集: ", args.data)
                print("生成模型: ", args.generate_model)
                print("受害模型: ", args.victim_model)
                print("攻击方法: ", args.kind)
                print("攻击算法: ", args.attack_algo)
                print("epsilon: ", args.epsilon)
                print("alpha_times: ", args.alpha_times)
                print("epoch: ", args.epoch)
                if args.is_training == 1:
                    Exp.attack()
                else:
                    Exp.load_attack()
    else:
        match args.task_name:
            case 'long_term_forecast':
                Exp = Exp_Long_Term_Forecast
            case 'classify_direction':
                Exp = Exp_Classify_Direction
            case _:
                raise ValueError('未能找到支持的任务名')

        if args.is_training == 1:
            for ii in range(args.itr):
                # if ii != 0:
                #     continue
                # 设置随机种子
                current_seed = ii
                set_seed(current_seed)
                # 不建议直接修改 args.seed，因为后面可能还要用到原始 seed
                # 但为了让 Exp 内部能拿到，我们暂且这样，或者传入一个新的 args
                args.seed = current_seed

                # setting record of experiments
                exp = Exp(args)  # set experiments
                # 将 seed 加入 setting 以区分不同次训练的模型
                setting = '{}_{}_s{}'.format(args.model_id, args.model, current_seed)

                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
        else:
            for ii in range(args.itr):
                # if ii != 0:
                #     continue
                # 设置随机种子
                current_seed = ii
                set_seed(current_seed)
                args.seed = current_seed

                exp = Exp(args)  # set experiments
                setting = '{}_{}_s{}'.format(args.model_id, args.model, current_seed)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, test=1)
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
