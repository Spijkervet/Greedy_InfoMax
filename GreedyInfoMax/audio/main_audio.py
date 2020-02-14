import os
import argparse
import torch
import time
import numpy as np

try:
    import hydra
    hydra_available = True
except ImportError:
    hydra_available = False



#### own modules
from GreedyInfoMax.utils import logger
from GreedyInfoMax.audio.arg_parser import arg_parser
from GreedyInfoMax.audio.models import load_audio_model
from GreedyInfoMax.audio.data import get_dataloader
from GreedyInfoMax.audio.validation import val_by_latent_speakers
from GreedyInfoMax.audio.validation import val_by_InfoNCELoss


def train(args, logs, model, optimizer):

    # get datasets and dataloaders
    (
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    ) = get_dataloader.get_libri_dataloaders(args)


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

        logs.create_log(model, epoch=epoch, optimizer=optimizer)


if hydra_available:
    @hydra.main(config_path=os.path.join(os.getcwd(), "config/audio/config.yaml"), strict=False)
    def hydra_main(cfg):
        args = argparse.Namespace(**cfg)
        # check_generic_args(cfg)
        # config = cfg.to_container()
        main(args)

def main(args):
    # Set start time
    args.time = time.ctime()

    # Device configuration
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Experiment
    args.experiment = "audio"

    arg_parser.create_log_path(args)

    # set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # initialize logger
    logs = logger.Logger(args)

    # load model
    model, optimizer = load_audio_model.load_model_and_optimizer(args)

    try:
        # Train the model
        train(args, logs, model, optimizer)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(model)


if __name__ == "__main__":
    if hydra_available:
        hydra_main()
    else:
        args = arg_parser.parse_args()
        main(args)
