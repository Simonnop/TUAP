export CUDA_VISIBLE_DEVICES=1,2

generate_model_name=ANN
victim_model_name=ANN

# generate_model_name=LSTM
# victim_model_name=LSTM

# data_path='./data/processed/eodp.csv'
data_path='./data/processed/spain-d3.csv'

python -u ./run_attack.py \
    --generate_model $generate_model_name \
    --victim_model $victim_model_name \
    --data 'Spain' \
    --kind 'first_come' \
    --attack_algo 'FGSM' \
    --time_window 24 \
    --epsilon 0.1 \
    --attack_rate 1 \
    --data_path $data_path \
    --device 'cpu'

exit 0

epsilon_list=(0.1 0.2 0.3)
# epsilon_list=(0.2 0.3 0.4)
# attack_rates=(0.5 0.75)
# epsilon_list=(0.1)
# attack_rates=(1.0)
attack_rates=(0.4 0.8)

# generate_model_name=Transformer
# victim_model_name=Transformer

for epsilon in "${epsilon_list[@]}"; do
  for attack_rate in "${attack_rates[@]}"; do
    python -u ./run_attack.py \
    --generate_model $generate_model_name \
    --victim_model $victim_model_name \
    --attack_algo 'FGSM' \
    --data 'Spain' \
    --kind 'first_come' \
    --time_window 24 \
    --epsilon $epsilon \
    --attack_rate $attack_rate \
    --data_path $data_path \
    --device 'cpu'
  done
done

# exit 0

generate_model_name=LSTM
victim_model_name=LSTM

for epsilon in "${epsilon_list[@]}"; do
  for attack_rate in "${attack_rates[@]}"; do
    python -u ./run_attack.py \
    --generate_model $generate_model_name \
    --victim_model $victim_model_name \
    --attack_algo 'FGSM' \
    --data 'Spain' \
    --kind 'first_come' \
    --time_window 24 \
    --epsilon $epsilon \
    --attack_rate $attack_rate \
    --data_path $data_path \
    --device 'cpu'
  done
done