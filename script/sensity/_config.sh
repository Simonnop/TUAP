# 灵敏度分析公共配置
# 在 script/sensity/ 下的各灵敏度脚本中 source 此文件

# 统一使用 ETTh1 数据集
# 生成模型列表, 受害模型: 四个都要
# generate_models=('SegRNN' 'iTransformer' 'TimesNet' 'FreTS')
generate_models=('SegRNN' 'iTransformer')
victim_models=('SegRNN' 'iTransformer' 'TimesNet' 'FreTS')
datasets=(ETTh1)

# 基础参数
base_epsilon=0.1
base_alpha_times=1
base_epoch=3
base_L=96
base_H=96
base_mu=1.0

# 所有攻击方法
# attack_algos=('FGSM' 'MIFGSM' 'BIM' 'PGD' 'TCA' 'ATSG' 'ADJM' 'BO' 'GGAA' 'GTW')
# attack_algos=('GTW')
attack_algos=('GTW_Fix')

# 单步攻击 (仅分析 epsilon, L, H)
attacks_single_step=('FGSM' 'ADJM' 'ATSG' 'BO' 'GTW' 'GTW_Fix')

# 多步攻击 (分析 epsilon, L, H, epoch, alpha_times)
attacks_multi_step=('MIFGSM' 'BIM' 'PGD' 'TCA' 'GGAA')

# MI 系列 (额外分析 mu)
attacks_mi=('MIFGSM' 'GGAA')

# 根据数据集名获取配置
get_dataset_config() {
    local ds=$1
    case $ds in
        ETTh1|ETTh2|ETTm1|ETTm2)
            root_path="./dataset/ETT-small/"
            data_path="${ds}.csv"
            data_name=$ds
            model_id_prefix=$ds
            enc_in=7
            dec_in=7
            c_out=7
            seg_len=24
            ;;
        ECL)
            root_path="./dataset/electricity/"
            data_path="electricity.csv"
            data_name="ECL"
            model_id_prefix="ECL"
            enc_in=321
            dec_in=321
            c_out=321
            seg_len=24
            ;;
        Exchange)
            root_path="./dataset/exchange_rate/"
            data_path="exchange_rate.csv"
            data_name="custom"
            model_id_prefix="Exchange"
            enc_in=8
            dec_in=8
            c_out=8
            seg_len=24
            ;;
        Weather)
            root_path="./dataset/weather/"
            data_path="weather.csv"
            data_name="custom"
            model_id_prefix="weather"
            enc_in=21
            dec_in=21
            c_out=21
            seg_len=48
            ;;
        ILI)
            root_path="./dataset/illness/"
            data_path="national_illness.csv"
            data_name="ILI"
            model_id_prefix="ILI"
            enc_in=7
            dec_in=7
            c_out=7
            seg_len=24
            ;;
        Traffic)
            root_path="./dataset/traffic/"
            data_path="traffic.csv"
            data_name="Traffic"
            model_id_prefix="Traffic"
            enc_in=862
            dec_in=862
            c_out=862
            seg_len=24
            ;;
        *)
            echo "Unknown dataset: $ds"
            return 1
            ;;
    esac
}
