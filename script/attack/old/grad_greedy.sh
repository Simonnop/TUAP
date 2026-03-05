
export CUDA_VISIBLE_DEVICES=1,2


data_path='./data/processed/spain-d3.csv'
sort_by='abs'

generate_model_name=ANN
victim_model_name=ANN

python -u ./run_attack.py \
      --generate_model $generate_model_name \
      --victim_model $victim_model_name \
      --kind 'grad_greedy' \
      --sort_by $sort_by \
      --data 'Spain' \
      --time_window 24 \
      --epsilon 0.1 \
      --attack_rate 0.4 \
      --seed 4060 \
      --data_path $data_path \
      --device 'cpu'

exit 0

epsilon_list=(0.1 0.2 0.3)
# epsilon_list=(0.2 0.3 0.4)
# attack_rates=(0.5 0.75)
attack_rates=(0.4 0.8)

generate_model_name=ANN
victim_model_name=ANN

for epsilon in "${epsilon_list[@]}"; do
  for attack_rate in "${attack_rates[@]}"; do
    python -u ./run_attack.py \
      --generate_model $generate_model_name \
      --victim_model $victim_model_name \
      --kind 'grad_greedy' \
      --sort_by $sort_by \
      --data 'Spain' \
      --time_window 24 \
      --epsilon $epsilon \
      --attack_rate $attack_rate \
      --seed 0 \
      --data_path $data_path \
      --device 'cpu'
  done
done

generate_model_name=LSTM
victim_model_name=LSTM

for epsilon in "${epsilon_list[@]}"; do
  for attack_rate in "${attack_rates[@]}"; do
    python -u ./run_attack.py \
      --generate_model $generate_model_name \
      --victim_model $victim_model_name \
      --kind 'grad_greedy' \
      --sort_by $sort_by \
      --data 'Spain' \
      --time_window 24 \
      --epsilon $epsilon \
      --attack_rate $attack_rate \
      --seed 0 \
      --data_path $data_path \
      --device 'cpu'
  done
done