export CUDA_VISIBLE_DEVICES=2

models=('iTransformer' 'TSMixer' 'SegRNN' 'TimesNet')

# 定义数据集数组
datasets=(ECL)

kinds=('wrong' 'global' 'first' 'random')

attack_algos=('FGSM')

# 基础参数设置
seq_len=96
label_len=48
pred_len=96
e_layers=2
d_layers=1
factor=3
features="M"
enc_in=321
dec_in=321
c_out=321

# 遍历每个数据集
for dataset in "${datasets[@]}"; do
    # 遍历每个模型
    for model_name in "${models[@]}"; do
        for kind in "${kinds[@]}"; do
            for attack_algo in "${attack_algos[@]}"; do
                echo "Training model: $model_name on dataset: $dataset"
                python -u run.py \
                    --task_name attack \
                    --is_training 1 \
                    --root_path ./dataset/electricity/ \
                    --data_path electricity.csv \
                    --model_id ${dataset}_${seq_len}_${pred_len} \
                    --generate_model $model_name \
                    --victim_model $model_name \
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
                    --des 'Exp' \
                    --itr 1

                echo "Finished training $model_name on $dataset"
                echo "----------------------------------------"
            done
        done
    done
done 
