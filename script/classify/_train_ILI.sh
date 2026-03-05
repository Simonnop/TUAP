export CUDA_VISIBLE_DEVICES=0

models=('iTransformer' 'SegRNN' 'TimesNet' 'FreTS')

# 定义数据集数组
datasets=(ILI)

attack_algos=('FGSM')

# 基础参数设置
seq_len=96
label_len=48
pred_len=96
e_layers=2
d_layers=1
factor=3
features="M"
enc_in=7
dec_in=7
c_out=7

# 遍历每个数据集
for dataset in "${datasets[@]}"; do
    # 遍历每个模型
    for model_name in "${models[@]}"; do
            opts=""
            if [ "$model_name" == "FreTS" ]; then
              opts="--channel_independence 1"
            fi
            for attack_algo in "${attack_algos[@]}"; do
                echo "Training model: $model_name on dataset: $dataset"
                python -u run.py \
                    --task_name classify_direction \
                    --is_training 1 \
                    --root_path ./dataset/illness/ \
                    --data_path national_illness.csv \
                    --model_id ${dataset}_${seq_len}_${pred_len} \
                    --generate_model $model_name \
                    --victim_model $model_name \
                    --epsilon 0.1 \
                    --data $dataset \
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
                    --des 'Exp' \
                    --itr 5 \
                    $opts

                echo "Finished training $model_name on $dataset"
                echo "----------------------------------------"
        done
    done
done 