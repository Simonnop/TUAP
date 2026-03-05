export CUDA_VISIBLE_DEVICES=1

generate_model_name=ANN
victim_model_name=ANN

data_path='./data/processed/spain-d3.csv'

python -u ./run_attack.py \
      --generate_model $generate_model_name \
      --victim_model $victim_model_name \
      --kind 'random' \
      --data 'Spain' \
      --time_window 24 \
      --epsilon 0.2 \
      --attack_rate 0.8 \
      --seed 4060 \
      --pop_size 30 \
      --max_iterations 200 \
      --max_early_stop 25 \
      --data_path $data_path \
      --device 'cpu'

# python -u /home/suruixian/experiments/RL-Attack-main/run_attack.py \
#       --generate_model $generate_model_name \
#       --victim_model $victim_model_name \
#       --action_type 'optimize' \
#       --mask_type 'optimize' \
#       --data 'Spain' \
#       --time_window 24 \
#       --epsilon 0.2 \
#       --attack_rate 0.5 \
#       --seed 101 \
#       --pop_size 10 \
#       --max_iterations 10 \
#       --max_early_stop 10 \
#       --data_path $data_path
