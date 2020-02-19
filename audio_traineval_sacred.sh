#!/bin/sh

echo "Using Sacred to log experiments\n"
echo "Training the Greedy InfoMax Model on audio data (librispeech)"
python -m GreedyInfoMax.audio.main_audio --name greedy_infomax_audio \
    with \
    data_input_dir=./datasets \
    subsample=True \
    num_epochs=300 \
    learning_rate=2e-4 \
    start_epoch=0 \
    output_data_dir=. \
    fp16=True \
    fp16_opt_level="O2"


# echo "Testing the Greedy InfoMax Model for phone classification"
# python -m GreedyInfoMax.audio.linear_classifiers.logistic_regression_phones \
#     --name logistic_regression_phones \
#     with \
#     model_path=./logs/1 \
#     model_num=299 \
#     data_input_dir=./datasets \
#     data_output_dir=. \
#     model_splits=1 # cpc

# echo "Testing the Greedy InfoMax Model for speaker classification"
# python -m GreedyInfoMax.audio.linear_classifiers.logistic_regression_speaker \
#     --name logistic_regression_speaker \
#     with \
#     model_path=/home/jspijkervet/git/Greedy_InfoMax/outputs/2020-02-14/18-23-37/logs/audio_experiment \
#     model_num=299 \
#     data_input_dir=./datasets \
#     data_output_dir=. \
#     model_splits=1 # cpc