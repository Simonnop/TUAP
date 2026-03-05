
models=('iTransformer' 'SegRNN' 'TimesNet' 'FreTS')

# 定义数据集数组
datasets=(Exchange)

attack_algos=('FGSM')

# 基础参数设置
# seq_len=96
label_len=48
# pred_len=96
# len=(48 72 96 120)
len=(96)
e_layers=2
d_layers=1
factor=3
features="M"
enc_in=8
dec_in=8
c_out=8

# 遍历每个数据集
for dataset in "${datasets[@]}"; do
    # 遍历每个模型
    for model_name in "${models[@]}"; do
            opts=""
            if [ "$model_name" == "FreTS" ]; then
              opts="--channel_independence 1"
            fi
            for attack_algo in "${attack_algos[@]}"; do
                for len in "${len[@]}"; do
                echo "Training model: $model_name on dataset: $dataset"
                python -u run.py \
                    --task_name classify_direction \
                    --is_training 1 \
                    --root_path ./dataset/exchange_rate/ \
                    --data_path exchange_rate.csv \
                    --model_id ${dataset}_${len}_${len} \
                    --generate_model $model_name \
                    --victim_model $model_name \
                    --epsilon 0.1 \
                    --data $dataset \
                    --attack_algo $attack_algo \
                    --features $features \
                    --seq_len $len \
                    --label_len $label_len \
                    --pred_len $len \
                    --e_layers $e_layers \
                    --d_layers $d_layers \
                    --factor $factor \
                    --enc_in $enc_in \
                    --dec_in $dec_in \
                    --c_out $c_out \
                    --des 'Exp' \
                    --itr 1 \
                    $opts

                echo "Finished training $model_name on $dataset"
                echo "----------------------------------------"
            done
        done
    done
done 