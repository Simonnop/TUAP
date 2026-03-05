export CUDA_VISIBLE_DEVICES=0

# Example script to run VMIFGSM and GGAA_VMIFGSM on ETTh1 dataset with TimesNet

seq_len=96
pred_len=96
model_name=TimesNet

# 1. Window-wise VMIFGSM
python -u run.py \
  --task_name attack \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --attack_algo VMIFGSM \
  --epsilon 0.05 \
  --epoch 10 \
  --generate_model $model_name \
  --victim_model $model_name 

# 2. GGAA with VMIFGSM
python -u run.py \
  --task_name attack \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --attack_algo GGAA_VMIFGSM \
  --epsilon 0.05 \
  --epoch 10 \
  --generate_model $model_name \
  --victim_model $model_name 
