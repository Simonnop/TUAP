export CUDA_VISIBLE_DEVICES=0
# # ETTh
bash script/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# bash script/long_term_forecast/ETT_script/TimesNet_ETTh2.sh
bash script/long_term_forecast/ETT_script/iTransformer_ETTh1.sh
# bash script/long_term_forecast/ETT_script/iTransformer_ETTh2.sh
bash script/long_term_forecast/ETT_script/SegRNN_ETTh1.sh
# bash script/long_term_forecast/ETT_script/SegRNN_ETTh2.sh
bash script/long_term_forecast/ETT_script/FreTS_ETTh1.sh
# bash script/long_term_forecast/ETT_script/FreTS_ETTh2.sh

# # # Traffic
# bash script/long_term_forecast/Traffic_script/TimesNet.sh
# bash script/long_term_forecast/Traffic_script/iTransformer.sh
# bash script/long_term_forecast/Traffic_script/SegRNN.sh
# bash script/long_term_forecast/Traffic_script/FreTS.sh

# # # Electricity
bash script/long_term_forecast/ECL_script/TimesNet.sh
bash script/long_term_forecast/ECL_script/iTransformer.sh
bash script/long_term_forecast/ECL_script/SegRNN.sh
bash script/long_term_forecast/ECL_script/FreTS.sh

# # # Exchange
# bash script/long_term_forecast/Exchange_script/TimesNet.sh
# bash script/long_term_forecast/Exchange_script/iTransformer.sh
# bash script/long_term_forecast/Exchange_script/SegRNN.sh
# bash script/long_term_forecast/Exchange_script/FreTS.sh

# # # ILI
# bash script/long_term_forecast/ILI_script/TimesNet.sh
# bash script/long_term_forecast/ILI_script/iTransformer.sh
# bash script/long_term_forecast/ILI_script/SegRNN.sh
# bash script/long_term_forecast/ILI_script/FreTS.sh