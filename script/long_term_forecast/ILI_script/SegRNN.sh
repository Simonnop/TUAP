
model_name=SegRNN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_96_96 \
  --model $model_name \
  --data ILI \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --seg_len 24 \
  --enc_in 7 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 5
