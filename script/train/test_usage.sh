export CUDA_VISIBLE_DEVICES=1

models=('iTransformer')

# 定义数据集数组
datasets=(ETTh1)

kinds=('classify_direction')

attack_algos=('MIFGSM')

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
    for model_name1 in "${models[@]}"; do
        for model_name2 in "${models[@]}"; do
            for kind in "${kinds[@]}"; do
                for attack_algo in "${attack_algos[@]}"; do
                    python -u run.py \
                        --task_name classify_direction \
                        --is_training 1 \
                        --root_path ./dataset/ETT-small/ \
                        --data_path ETTh1.csv \
                        --model_id ${dataset}_${seq_len}_${pred_len} \
                        --generate_model $model_name1 \
                        --victim_model $model_name2 \
                        --epsilon 0.1 \
                        --data $dataset \
                        --kind $kind \
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
                        --use_gpu 1 \
                        --des 'Exp' \
                        --itr 1
                done
            done
        done
    done
done