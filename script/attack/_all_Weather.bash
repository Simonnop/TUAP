export CUDA_VISIBLE_DEVICES=1

# models=('TSMixer' 'SegRNN' 'PatchTST')
models=('iTransformer' 'SegRNN' 'TimesNet' 'FreTS')

# 定义数据集数组
datasets=(Weather)

# kinds=('wrong' 'first' 'random' 'mask' 'optimize' 'classify' 'global')

attack_algos=('FGSM' 'MIFGSM' 'BIM' 'PGD' 'TCA' 'ATSG' 'AAIM' 'ADJM' 'BO' 'GGAA')

# epsilons=(0.05 0.1 0.15 0.2)
epsilons=(0.1)

# alpha_times_list=(1 5 10 15)
alpha_times_list=(1)

# epoch_list=(1 5 10 15)
epoch_list=(10)

itr=1

# 基础参数设置
seq_len=96
label_len=48
pred_len=96
e_layers=2
d_layers=1
factor=3
features="M"
enc_in=21
dec_in=21
c_out=21

# 遍历每个数据集
for dataset in "${datasets[@]}"; do
    # 遍历每个模型
    for model_name1 in "${models[@]}"; do
        for model_name2 in "${models[@]}"; do
                for attack_algo in "${attack_algos[@]}"; do
                    for epsilon in "${epsilons[@]}"; do
                        for alpha_times in "${alpha_times_list[@]}"; do
                            for epoch in "${epoch_list[@]}"; do
                    python -u run.py \
                        --task_name attack \
                        --is_training 0 \
                        --root_path ./dataset/weather/ \
                        --data_path weather.csv \
                        --model_id ${dataset}_${seq_len}_${pred_len} \
                        --generate_model $model_name1 \
                        --victim_model $model_name2 \
                        --epsilon $epsilon \
                        --alpha_times $alpha_times \
                        --epoch $epoch \
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
                        --use_gpu 1 \
                        --des 'Exp' \
                        --itr $itr \
                        --record_name 'attack'
                        done
                    done
                done
            done
        done
    done
done