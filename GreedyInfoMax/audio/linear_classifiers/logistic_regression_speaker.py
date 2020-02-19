import sys
import os
import argparse
import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

try:
    from sacred import Experiment
    from sacred.stflow import LogFileWriter
    from sacred.observers import FileStorageObserver, MongoObserver

    sacred_available = True

    #### pass configuration
    ex = Experiment("logistic_regression_phones")

    #### file output directory
    ex.observers.append(FileStorageObserver("./logs"))

    #### database output, make sure to configure the right user
    ex.observers.append(
        MongoObserver().create(
            url=f"mongodb://admin:admin@localhost:27017/?authMechanism=SCRAM-SHA-1",
            db_name="db",
        )
    )


except Exception as e:
    sacred_available = False
    print(e)


## own modules
from GreedyInfoMax.audio.data import get_dataloader
from GreedyInfoMax.utils import logger
from GreedyInfoMax.audio.arg_parser import arg_parser
from GreedyInfoMax.audio.models import load_audio_model, loss_supervised_speaker


def train(opt, context_model, loss, train_loader, optimizer, writer, logs):
    total_step = len(train_loader)
    print_idx = 100

    total_i = 0
    for epoch in range(opt.num_epochs):
        loss_epoch = 0
        acc_epoch = 0
        for i, (audio, filename, _, audio_idx) in enumerate(train_loader):

            starttime = time.time()

            loss.zero_grad()

            ### get latent representations for current audio
            model_input = audio.to(opt.device)

            with torch.no_grad():
                z = context_model.module.forward_through_n_layers(model_input, 5)

            z = z.detach()

            # forward pass
            total_loss, accuracies = loss.get_loss(
                model_input, z, z, filename, audio_idx
            )

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            sample_loss = total_loss.item()
            accuracy = accuracies.item()

            writer.add_scalar("Loss/train_step", sample_loss, total_i)
            writer.add_scalar("Accuracy/train_step", accuracy, total_i)
            writer.flush()

            if i % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Accuracy: {:.4f}, Loss: {:.4f}".format(
                        epoch + 1,
                        opt.num_epochs,
                        i,
                        total_step,
                        time.time() - starttime,
                        accuracy,
                        sample_loss,
                    )
                )
                starttime = time.time()

            loss_epoch += sample_loss
            acc_epoch += accuracy
            total_i += 1

        logs.append_train_loss([loss_epoch / total_step])
        writer.add_scalar("Loss/train_epoch", loss_epoch / total_step, epoch)
        writer.add_scalar("Accuracy/train_epoch", acc_epoch / total_step, epoch)
        writer.flush()

        # Sacred
        ex.log_scalar("train.loss", loss_epoch / total_step)
        ex.log_scalar("train.accuracy", acc_epoch / total_step)


def test(opt, context_model, loss, data_loader):
    loss.eval()

    accuracy = 0
    loss_epoch = 0

    with torch.no_grad():
        for i, (audio, filename, _, audio_idx) in enumerate(data_loader):

            loss.zero_grad()

            ### get latent representations for current audio
            model_input = audio.to(opt.device)

            with torch.no_grad():
                z = context_model.module.forward_through_n_layers(model_input, 5)

            z = z.detach()

            # forward pass
            total_loss, step_accuracy = loss.get_loss(
                model_input, z, z, filename, audio_idx
            )

            accuracy += step_accuracy.item()
            loss_epoch += total_loss.item()

            if i % 10 == 0:
                print(
                    "Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}".format(
                        i, len(data_loader), loss_epoch / (i + 1), accuracy / (i + 1)
                    )
                )

    accuracy = accuracy / len(data_loader)
    loss_epoch = loss_epoch / len(data_loader)
    print("Final Testing Accuracy: ", accuracy)
    print("Final Testing Loss: ", loss_epoch)
    return loss_epoch, accuracy


def main(opt, experiment_name):
    opt.time = time.ctime()

    # Device configuration
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt.batch_size = 64
    opt.num_epochs = 50
    opt.learning_rate = 1e-3

    arg_parser.create_log_path(opt, add_path_var="linear_model")

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    ## load model
    context_model, optimizer = load_audio_model.load_model_and_optimizer(
        opt, reload_model=True, calc_accuracy=True, num_GPU=1,
    )
    context_model.eval()

    n_features = context_model.module.reg_hidden

    loss = loss_supervised_speaker.Speaker_Loss(opt, n_features, calc_accuracy=True)

    optimizer = torch.optim.Adam(loss.parameters(), lr=opt.learning_rate)

    # load dataset
    train_loader, _, test_loader, _ = get_dataloader.get_libri_dataloaders(opt)

    logs = logger.Logger(opt)
    accuracy = 0

    # set comment to experiment's name
    tb_dir = os.path.join(opt.log_path, experiment_name)
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    try:
        # Train the model
        train(opt, context_model, loss, train_loader, optimizer, writer, logs)

        # Test the model
        result_loss, accuracy = test(opt, context_model, loss, test_loader)

    except KeyboardInterrupt:
        print("Training interrupted, saving log files")

    logs.create_log(loss, accuracy=accuracy, final_test=True, final_loss=result_loss)


if sacred_available:
    from GreedyInfoMax.utils.yaml_config_hook import yaml_config_hook

    @ex.config
    def my_config():
        yaml_config_hook("./config/audio/config.yaml", ex)

        #### override any settings here
        # start_epoch = 100
        # ex.add_config(
        #   {'start_epoch': start_epoch})

    @ex.automain
    @LogFileWriter(ex)
    def sacred_main(_run, _log):
        args = argparse.Namespace(**_run.config)
        if len(_run.observers) > 1:
            out_dir = _run.observers[1].dir
        else:
            out_dir = _run.observers[0].dir

        # set the log dir
        args.out_dir = out_dir
        args.use_sacred = True
        main(args, experiment_name=_run.experiment_info["name"])


if __name__ == "__main__":
    if not sacred_available:
        args = arg_parser.parse_args()
        args.use_sacred = False
        main(args, experiment_name="logistic_regression_speaker")

