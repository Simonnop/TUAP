export CUDA_VISIBLE_DEVICES=1

models=('iTransformer' 'TSMixer' 'SegRNN' 'TimesNet')

# 定义数据集数组
datasets=(Weather)

len=(48 72 96 120)
# 基础参数设置
label_len=48
e_layers=3
d_layers=1
factor=3
features="M"
enc_in=21
dec_in=21
c_out=21
itr=10

# 遍历每个数据集
for dataset in "${datasets[@]}"; do
    # 遍历每个模型
    for model_name in "${models[@]}"; do
        for len_val in "${len[@]}"; do
            echo "Training model: $model_name on dataset: $dataset with length $len_val"
            
            # 根据模型设置不同的参数
            if [ "$model_name" == "xLSTM" ]; then
                python -u run.py \
                  --task_name long_term_forecast \
                  --is_training 1 \
                  --root_path ./dataset/weather/ \
                  --data_path weather.csv \
                  --model_id ${dataset}_${len_val}_${len_val} \
                  --model $model_name \
                  --data $dataset \
                  --features $features \
                  --seq_len $len_val \
                  --label_len $label_len \
                  --pred_len $len_val \
                  --e_layers $e_layers \
                  --d_layers $d_layers \
                  --factor $factor \
                  --enc_in $enc_in \
                  --dec_in $dec_in \
                  --c_out $c_out \
                  --des 'Exp' \
                  --d_model 128 \
                  --d_ff 256 \
                  --itr $itr
            elif [ "$model_name" == "iTransformer" ]; then
                python -u run.py \
                  --task_name long_term_forecast \
                  --is_training 1 \
                  --root_path ./dataset/weather/ \
                  --data_path weather.csv \
                  --model_id ${dataset}_${len_val}_${len_val} \
                  --model $model_name \
                  --data $dataset \
                  --features $features \
                  --seq_len $len_val \
                  --label_len $label_len \
                  --pred_len $len_val \
                  --e_layers $e_layers \
                  --d_layers $d_layers \
                  --factor $factor \
                  --enc_in $enc_in \
                  --dec_in $dec_in \
                  --c_out $c_out \
                  --des 'Exp' \
                  --d_model 512 \
                  --d_ff 512 \
                  --itr $itr
            elif [ "$model_name" == "TSMixer" ]; then
                python -u run.py \
                  --task_name long_term_forecast \
                  --is_training 1 \
                  --root_path ./dataset/weather/ \
                  --data_path weather.csv \
                  --model_id ${dataset}_${len_val}_${len_val} \
                  --model $model_name \
                  --data $dataset \
                  --features $features \
                  --seq_len $len_val \
                  --label_len $label_len \
                  --pred_len $len_val \
                  --e_layers 2 \
                  --d_layers $d_layers \
                  --factor $factor \
                  --enc_in $enc_in \
                  --dec_in $dec_in \
                  --c_out $c_out \
                  --d_model 32 \
                  --d_ff 32 \
                  --top_k 5 \
                  --des 'Exp' \
                  --itr $itr
            elif [ "$model_name" == "SegRNN" ]; then
                python -u run.py \
                  --task_name long_term_forecast \
                  --is_training 1 \
                  --root_path ./dataset/weather/ \
                  --data_path weather.csv \
                  --model_id ${dataset}_${len_val}_${len_val} \
                  --model $model_name \
                  --data $dataset \
                  --features $features \
                  --seq_len $len_val \
                  --pred_len $len_val \
                  --seg_len 48 \
                  --enc_in $enc_in \
                  --d_model 512 \
                  --dropout 0.5 \
                  --learning_rate 0.0001 \
                  --des 'Exp' \
                  --itr $itr
            elif [ "$model_name" == "TimesNet" ]; then
                python -u run.py \
                  --task_name long_term_forecast \
                  --is_training 1 \
                  --root_path ./dataset/weather/ \
                  --data_path weather.csv \
                  --model_id ${dataset}_${len_val}_${len_val} \
                  --model $model_name \
                  --data $dataset \
                  --features $features \
                  --seq_len $len_val \
                  --label_len $label_len \
                  --pred_len $len_val \
                  --e_layers 2 \
                  --d_layers $d_layers \
                  --factor $factor \
                  --enc_in $enc_in \
                  --dec_in $dec_in \
                  --c_out $c_out \
                  --d_model 32 \
                  --d_ff 32 \
                  --top_k 5 \
                  --des 'Exp' \
                  --itr $itr
            fi
            
            echo "Finished training $model_name on $dataset with length $len_val"
            echo "----------------------------------------"
        done
    done
done