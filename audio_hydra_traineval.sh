#!/bin/sh

# echo "Training the Greedy InfoMax Model on audio data (librispeech)"
# python -m GreedyInfoMax.audio.main_audio \
#     data_input_dir=/storage/jspijkervet/Greedy_InfoMax/datasets \
#     subsample=True \
#     num_epochs=1000 \
#     learning_rate=2e-4 \
#     start_epoch=0 \
#     output_data_dir=. \
#     save_dir=audio_experiment

# echo "Training the Greedy InfoMax Model on audio data (librispeech) with DeepSpeed"
# deepspeed.pt GreedyInfoMax/audio/main_audio.py \
#     deepspeed_config=/home/deepspeed/Greedy_InfoMax/deepspeed/ds_config.json \
#     data_input_dir=/home/deepspeed/Greedy_InfoMax/datasets \
#     subsample=True \
#     num_epochs=300 \
#     learning_rate=2e-4 \
#     start_epoch=0 \
#     output_data_dir=. \
#     save_dir=audio_experiment


echo "Testing the Greedy InfoMax Model for phone classification"
python -m GreedyInfoMax.audio.linear_classifiers.logistic_regression_phones \
    model_path=/home/jspijkervet/git/Greedy_InfoMax/outputs/2020-02-14/18-23-37/logs/audio_experiment \
    model_num=299 \
    data_input_dir=/storage/jspijkervet/Greedy_InfoMax/datasets \
    data_output_dir=. \
    model_splits=1 # cpc

echo "Testing the Greedy InfoMax Model for speaker classification"
python -m GreedyInfoMax.audio.linear_classifiers.logistic_regression_speaker \
    model_path=/home/jspijkervet/git/Greedy_InfoMax/outputs/2020-02-14/18-23-37/logs/audio_experiment \
    model_num=299 \
    data_input_dir=/storage/jspijkervet/Greedy_InfoMax/datasets \
    data_output_dir=. \
    model_splits=1 # cpc