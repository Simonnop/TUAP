export CUDA_VISIBLE_DEVICES=1,2

victim_model_name=ANN

data_path='./data/processed/spain-d3.csv'

# python -u ./run_attack.py \
#     --victim_model $victim_model_name \
#     --data 'Spain' \
#     --kind 'optimize_delta' \
#     --time_window 24 \
#     --epsilon 0.2 \
#     --attack_rate 0.8 \
#     --pop_size 30 \
#     --max_iterations 200 \
#     --max_early_stop 25 \
#     --data_path $data_path \
#     --device 'cpu'

epsilon_list=(0.1 0.2 0.3)
# epsilon_list=(0.2 0.3 0.4)
# attack_rates=(0.5 0.75)
attack_rates=(0.4 0.8)

for epsilon in "${epsilon_list[@]}"; do
  for attack_rate in "${attack_rates[@]}"; do
    python -u ./run_attack.py \
    --victim_model $victim_model_name \
    --data 'Spain' \
    --kind 'optimize_delta' \
    --time_window 24 \
    --epsilon $epsilon \
    --attack_rate $attack_rate \
    --pop_size 30 \
    --max_iterations 200 \
    --max_early_stop 25 \
    --data_path $data_path \
    --device 'cpu'
  done
done

victim_model_name=LSTM

for epsilon in "${epsilon_list[@]}"; do
  for attack_rate in "${attack_rates[@]}"; do
    python -u ./run_attack.py \
    --victim_model $victim_model_name \
    --data 'Spain' \
    --kind 'optimize_delta' \
    --time_window 24 \
    --epsilon $epsilon \
    --attack_rate $attack_rate \
    --pop_size 30 \
    --max_iterations 200 \
    --max_early_stop 25 \
    --data_path $data_path \
    --device 'cuda'
  done
done