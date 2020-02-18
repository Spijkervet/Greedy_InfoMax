#!/bin/sh

echo "Training the Greedy InfoMax Model on vision data (stl-10)"
python -m GreedyInfoMax.vision.main_vision --download_dataset --batch_size 16 --learning_rate 1.5e-4 --num_epochs 300 --save_dir vision_experiment

# echo "Testing the Greedy InfoMax Model for image classification"
# python -m GreedyInfoMax.vision.downstream_classification --model_path ./logs/vision_experiment --model_num 299