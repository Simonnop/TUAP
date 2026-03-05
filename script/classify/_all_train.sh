export CUDA_VISIBLE_DEVICES=1

bash script/classify/_train_ETT.sh
# bash script/classify/_train_Exchange.sh
# bash script/classify/_train_ILI.sh
# bash script/classify/_train_Weather.sh
bash script/classify/_train_ECL.sh
# bash script/classify/_train_Traffic.sh