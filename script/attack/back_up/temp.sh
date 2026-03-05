
export CUDA_VISIBLE_DEVICES=1,2

generate_model_name=ANN
victim_model_name=ANN

data_path='./data/processed/spain-d3.csv'

python -u ./run_attack.py \
    --generate_model $generate_model_name \
    --victim_model $victim_model_name \
    --kind 'grad_greedy' \
    --sort_by 'abs' \
    --data 'Spain' \
    --time_window 24 \
    --epsilon 0.2 \
    --attack_rate 0.8 \
    --seed 4060 \
    --data_path $data_path \
    --device 'cuda'