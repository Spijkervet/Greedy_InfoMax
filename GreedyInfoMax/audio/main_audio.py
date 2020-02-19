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
    ex = Experiment("greedy_infomax")

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


#### own modules
from GreedyInfoMax.utils import logger
from GreedyInfoMax.audio.arg_parser import arg_parser
from GreedyInfoMax.audio.models import load_audio_model
from GreedyInfoMax.audio.data import get_dataloader
from GreedyInfoMax.audio.validation import val_by_latent_speakers
from GreedyInfoMax.audio.validation import val_by_InfoNCELoss


def train(args, model, optimizer, writer, logs):

    # get datasets and dataloaders
    (
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    ) = get_dataloader.get_libri_dataloaders(args)

    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # model_engine, optimizer, trainloader, __ = deepspeed.initialize(args=args, model=model, model_parameters=parameters, training_data=train_dataset)

    total_step = len(train_loader)
    # how often to output training values
    print_idx = 100
    # how often to validate training process by plotting latent representations of various speakers
    latent_val_idx = 1000

    starttime = time.time()

    for epoch in range(args.start_epoch, args.num_epochs + args.start_epoch):

        loss_epoch = [0 for i in range(args.model_splits)]

        for step, (audio, filename, _, start_idx) in enumerate(train_loader):

            # validate training progress by plotting latent representation of various speakers
            if step % latent_val_idx == 0:
                val_by_latent_speakers.val_by_latent_speakers(
                    args, train_dataset, model, epoch, step
                )

            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}".format(
                        epoch + 1,
                        args.num_epochs + args.start_epoch,
                        step,
                        total_step,
                        time.time() - starttime,
                    )
                )

            starttime = time.time()

            model_input = audio.to(args.device)

            loss = model(model_input, filename, start_idx, n=args.train_layer)
            loss = torch.mean(loss, 0)  # average over the losses from different GPUs

            for idx, cur_losses in enumerate(loss):
                model.zero_grad()

                if idx == len(loss) - 1:
                    cur_losses.backward()
                else:
                    cur_losses.backward(retain_graph=True)
                optimizer[idx].step()

                print_loss = cur_losses.item()
                if step % print_idx == 0:
                    print("\t \t Loss: \t \t {:.4f}".format(print_loss))

                loss_epoch[idx] += print_loss

        logs.append_train_loss([x / total_step for x in loss_epoch])

        # validate by testing the CPC performance on the validation set
        if args.validate:
            validation_loss = val_by_InfoNCELoss.val_by_InfoNCELoss(
                args, model, test_loader
            )
            logs.append_val_loss(validation_loss)

        # log metrics
        mean_loss = np.array([x / total_step for x in loss_epoch]).mean()
        # Tensorboard
        writer.add_scalar("Loss/train", mean_loss, epoch)
        writer.flush()

        # Sacred
        ex.log_scalar("train.loss_epoch", mean_loss)

        logs.create_log(model, epoch=epoch, optimizer=optimizer)


def main(args, experiment_name):

    # set start time
    args.time = time.ctime()

    # Device configuration
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    arg_parser.create_log_path(args)

    # set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # initialize logger
    logs = logger.Logger(args)

    # load model
    model, optimizer = load_audio_model.load_model_and_optimizer(args)

    # set comment to experiment's name
    tb_dir = os.path.join(args.log_path, experiment_name)
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    try:
        # Train the model
        train(args, model, optimizer, writer, logs)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(model)


if sacred_available:
    from GreedyInfoMax.utils.yaml_config_hook import yaml_config_hook

    @ex.config
    def my_config():
        yaml_config_hook('./config/audio/config.yaml', ex)

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
        main(args, experiment_name="greedy_infomax")

