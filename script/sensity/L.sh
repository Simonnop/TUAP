#!/bin/bash
# 灵敏度分析: L (seq_len)

export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
source "$SCRIPT_DIR/_config.sh"

seq_lens=(48 72 96 120)
pred_len=$base_H
label_len=48
e_layers=2
d_layers=1
factor=3
features="M"
itr=1
record_name="sensi_L"

for dataset in "${datasets[@]}"; do
    get_dataset_config "$dataset" || continue

    for generate_model in "${generate_models[@]}"; do
    for victim_model in "${victim_models[@]}"; do
        for attack_algo in "${attack_algos[@]}"; do
            if [[ " ${attacks_single_step[*]} " =~ " ${attack_algo} " ]]; then
                use_epoch=1
                use_alpha_times=1
            else
                use_epoch=$base_epoch
                use_alpha_times=$base_alpha_times
            fi

            for seq_len in "${seq_lens[@]}"; do
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
                    --save_sample_delta 1 \
                    --record_name $record_name
            done
        done
    done
    done
done
