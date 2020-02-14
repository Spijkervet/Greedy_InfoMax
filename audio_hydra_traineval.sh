#!/bin/sh

echo "Training the Greedy InfoMax Model on audio data (librispeech)"
python -m GreedyInfoMax.audio.main_audio subsample=True num_epochs=300 learning_rate=2e-4 start_epoch=0 output_data_dir=. save_dir=audio_experiment

# echo "Testing the Greedy InfoMax Model for phone classification"
# python -m GreedyInfoMax.audio.linear_classifiers.logistic_regression_phones --model_path ./logs/audio_experiment --model_num 999 -i ./datasets/ -o .

# echo "Testing the Greedy InfoMax Model for speaker classification"
# python -m GreedyInfoMax.audio.linear_classifiers.logistic_regression_speaker --model_path ./logs/audio_experiment --model_num 999 -i ./datasets/ -o .
