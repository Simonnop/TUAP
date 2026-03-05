#!/bin/bash
# 灵敏度分析: L (seq_len)
# 仅 FGSM GTW GGAA 白盒攻击，统计每次运行时间写入 CSV

export CUDA_VISIBLE_DEVICES=2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# 内联配置，不引入外部 sh
datasets=(ETTh1)
victim_models=('SegRNN')
attack_algos=('FGSM' 'GTW' 'GGAA')
attacks_single_step=('FGSM' 'GTW')

base_epsilon=0.1
base_alpha_times=1
base_epoch=10
base_H=96

seq_lens=(48 72 96 120)
pred_len=$base_H
label_len=48
e_layers=2
d_layers=1
factor=3
features="M"
itr=1
record_name="attack_time_L"

# ETTh1 数据集配置
root_path="./dataset/ETT-small/"
data_path="ETTh1.csv"
data_name="ETTh1"
model_id_prefix="ETTh1"
enc_in=7
dec_in=7
c_out=7
seg_len=24

# 初始化时间统计 CSV
TIME_CSV="./time_L.csv"
echo "dataset,victim_model,attack_algo,seq_len,elapsed_seconds" > "$TIME_CSV"

for dataset in "${datasets[@]}"; do
    for victim_model in "${victim_models[@]}"; do
        # 白盒: generate_model == victim_model
        generate_model=$victim_model

        for attack_algo in "${attack_algos[@]}"; do
            if [[ " ${attacks_single_step[*]} " =~ " ${attack_algo} " ]]; then
                use_epoch=1
                use_alpha_times=1
            else
                use_epoch=$base_epoch
                use_alpha_times=$base_alpha_times
            fi

            for seq_len in "${seq_lens[@]}"; do
                start=$(date +%s.%N)
                python -u run.py \
                    --task_name attack \
                    --is_training 0 \
                    --root_path "$root_path" \
                    --data_path "$data_path" \
                    --model_id ${model_id_prefix}_${seq_len}_${pred_len} \
                    --generate_model $generate_model \
                    --victim_model $victim_model \
                    --epsilon $base_epsilon \
                    --alpha_times $use_alpha_times \
                    --epoch $use_epoch \
                    --data $data_name \
                    --attack_algo $attack_algo \
                    --features $features \
                    --seq_len $seq_len \
                    --label_len $label_len \
                    --pred_len $pred_len \
                    --e_layers $e_layers \
                    --d_layers $d_layers \
                    --factor $factor \
                    --enc_in $enc_in \
                    --dec_in $dec_in \
                    --c_out $c_out \
                    --seg_len $seg_len \
                    --use_gpu 1 \
                    --des 'Exp' \
                    --itr $itr \
                    --record_name $record_name
                end=$(date +%s.%N)
                elapsed=$(python3 -c "print($end - $start)")
                echo "$dataset,$victim_model,$attack_algo,$seq_len,$elapsed" >> "$TIME_CSV"
            done
        done
    done
done
