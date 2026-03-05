export CUDA_VISIBLE_DEVICES=1

models=('iTransformer' 'TSMixer' 'SegRNN' 'TimesNet')
# models=('TimesNet')

# 定义数据集数组
datasets=(ETTh1 ETTh2)

len=(48 72 96 120)
# len=(96)

# 基础参数设置
# seq_len=96
label_len=48
# pred_len=96
e_layers=2
d_layers=1
factor=3
features="M"
enc_in=7
dec_in=7
c_out=7

for len in "${len[@]}"; do
# 遍历每个数据集
for dataset in "${datasets[@]}"; do
    # 遍历每个模型
    for model_name in "${models[@]}"; do
        # 动态构建模型专用的额外参数
        if [ "$model_name" = "TimesNet" ]; then
            model_args="--d_model 16 --d_ff 32 --top_k 5"
        else
            # 其他模型不传递这些参数，使用默认值
            model_args=""
        fi

        echo "Training model: $model_name on dataset: $dataset"
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ${dataset}.csv \
          --model_id ${dataset}_${len}_${len} \
          --model $model_name \
          --data $dataset \
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
          --itr 10 \
          $model_args
        
          echo "Finished training $model_name on $dataset with length $len"
          echo "----------------------------------------"
      done
  done
done 