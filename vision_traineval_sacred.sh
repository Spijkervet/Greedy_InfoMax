#!/bin/sh

echo "Using Sacred to log experiments\n"
echo "Training the Greedy InfoMax Model on vision data (stl-10)"
python -m GreedyInfoMax.vision.main_vision --name vision_experiment \
    with \
    data_input_dir=./datasets \
    download_dataset=False \
    batch_size=16 \
    learning_rate=1.5e-4 \
    num_epochs=300 \
    fp16=True \
    fp16_opt_level="O2"

# echo "Testing the Greedy InfoMax Model for image classification"
# python -m GreedyInfoMax.vision.downstream_classification --model_path ./logs/vision_experiment --model_num 299