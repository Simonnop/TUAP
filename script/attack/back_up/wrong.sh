export CUDA_VISIBLE_DEVICES=1

generate_model_name=Transformer
victim_model_name=Transformer

# generate_model_name=LSTM
# victim_model_name=LSTM

python -u ./run.py \
    --task_name 'attack' \
    --generate_model $generate_model_name \
    --victim_model $victim_model_name \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --data 'ECL' \
    --kind 'wrong' \
    --attack_algo 'FGSM' \
    --features 'S' \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --epsilon 0.1 \
    --attack_rate 1 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --device 'cuda'

exit 0

epsilon_list=(0.1 0.2 0.3)
# epsilon_list=(0.2 0.3 0.4)
# attack_rates=(0.5 0.75)
attack_rates=(1.0)
# attack_rates=(0.4 0.8)

generate_model_name=Transformer
victim_model_name=Transformer

for epsilon in "${epsilon_list[@]}"; do
  for attack_rate in "${attack_rates[@]}"; do
    python -u ./run_attack.py \
    --generate_model $generate_model_name \
    --victim_model $victim_model_name \
    --attack_algo 'FGSM' \
    --data 'EODP' \
    --kind 'wrong' \
    --time_window 24 \
    --epsilon $epsilon \
    --attack_rate $attack_rate \
    --data_path $data_path \
    --device 'cpu'
  done
done

exit 0

generate_model_name=LSTM
victim_model_name=LSTM

for epsilon in "${epsilon_list[@]}"; do
  for attack_rate in "${attack_rates[@]}"; do
    python -u ./run_attack.py \
    --generate_model $generate_model_name \
    --victim_model $victim_model_name \
    --attack_algo 'FGSM' \
    --data 'EODP' \
    --kind 'wrong' \
    --time_window 24 \
    --epsilon $epsilon \
    --attack_rate $attack_rate \
    --data_path $data_path \
    --device 'cpu'
  done
done