
export CUDA_VISIBLE_DEVICES=1,2

data_path='./data/processed/eodp-d3.csv'

generate_model_name=ANN
victim_model_name=ANN

epsilon_list=(0.1 0.2 0.3)
attack_rates=(0.25 0.5 0.75 1.0)
# seeds=(1 2 3)  # 添加多个seed
seeds=(3)
# attack_algos=('PGD' 'FGSM' 'MIFGSM')  # 添加多个攻击算法
attack_algos=('FGSM')


python -u ./run_attack.py \
    --generate_model $generate_model_name \
    --victim_model $victim_model_name \
    --attack_algo 'FGSM' \
    --data 'EODP' \
    --kind 'global_local' \
    --time_window 24 \
    --epsilon 0.1 \
    --attack_rate 0.4 \
    --data_path $data_path \
    --seed 3 \
    --device 'cuda'

exit 0


for epsilon in "${epsilon_list[@]}"; do
  for attack_rate in "${attack_rates[@]}"; do
    for seed in "${seeds[@]}"; do  # 遍历多个seed
      for attack_algo in "${attack_algos[@]}"; do  # 遍历多个攻击算法
        python -u ./run_attack.py \
          --generate_model $generate_model_name \
          --victim_model $victim_model_name \
          --kind 'global_local' \
          --attack_algo $attack_algo \
          --data 'Spain' \
          --time_window 24 \
          --epsilon $epsilon \
          --attack_rate $attack_rate \
          --seed $seed \
          --data_path $data_path \
          --device 'cpu'
      done
    done
  done
done

generate_model_name=LSTM
victim_model_name=LSTM

for epsilon in "${epsilon_list[@]}"; do
  for attack_rate in "${attack_rates[@]}"; do
    for seed in "${seeds[@]}"; do  # 遍历多个seed
      for attack_algo in "${attack_algos[@]}"; do  # 遍历多个攻击算法
        python -u ./run_attack.py \
          --generate_model $generate_model_name \
          --victim_model $victim_model_name \
          --kind 'global_local' \
          --attack_algo $attack_algo \
          --data 'Spain' \
          --time_window 24 \
          --epsilon $epsilon \
          --attack_rate $attack_rate \
          --seed $seed \
          --data_path $data_path \
          --device 'cpu'
      done
    done
  done
done