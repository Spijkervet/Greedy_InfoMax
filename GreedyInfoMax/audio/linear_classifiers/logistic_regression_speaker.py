import sys
import os
import argparse
import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

## own modules
from GreedyInfoMax.audio.data import get_dataloader
from GreedyInfoMax.utils import logger
from GreedyInfoMax.audio.arg_parser import arg_parser
from GreedyInfoMax.audio.models import load_audio_model, loss_supervised_speaker

try:
    import hydra

    hydra_available = True
except ImportError:
    hydra_available = False


def train(opt, context_model, loss, train_loader, optimizer, logs):
    total_step = len(train_loader)
    writer = SummaryWriter(log_dir=os.path.join(os.getcwd()))
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
                z = context_model.module.forward_through_n_layers(
                    model_input, 5
                )

            z = z.detach()

            # forward pass
            total_loss, accuracies = loss.get_loss(model_input, z, z, filename, audio_idx)

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            sample_loss = total_loss.item()
            accuracy = accuracies.item()
            
            writer.add_scalar('Loss/train_step', sample_loss, total_i)
            writer.add_scalar('Accuracy/train_step', accuracy, total_i)
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
        writer.add_scalar('Loss/train_epoch', loss_epoch / total_step, epoch)
        writer.add_scalar('Accuracy/train_epoch', acc_epoch / total_step, epoch)
        writer.flush()


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
                z = context_model.module.forward_through_n_layers(
                    model_input, 5
                )

            z = z.detach()

            # forward pass
            total_loss, step_accuracy = loss.get_loss(model_input, z, z, filename, audio_idx)

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


if hydra_available:

    @hydra.main(
        config_path=os.path.join(os.getcwd(), "config/audio/config.yaml"), strict=False
    )
    def hydra_main(cfg):
        args = argparse.Namespace(**cfg)
        # check_generic_args(cfg)
        # config = cfg.to_container()
        main(args)

def main(opt):
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
        opt,
        reload_model=True,
        calc_accuracy=True,
        num_GPU=1,
    )
    context_model.eval()

    n_features = context_model.module.reg_hidden

    loss = loss_supervised_speaker.Speaker_Loss(
        opt, n_features, calc_accuracy=True
    )

    optimizer = torch.optim.Adam(loss.parameters(), lr=opt.learning_rate)

    # load dataset
    train_loader, _, test_loader, _ = get_dataloader.get_libri_dataloaders(opt)

    logs = logger.Logger(opt)
    accuracy = 0

    try:
        # Train the model
        train(opt, context_model, loss, train_loader, optimizer, logs)

        # Test the model
        result_loss, accuracy = test(opt, context_model, loss, test_loader)

    except KeyboardInterrupt:
        print("Training interrupted, saving log files")

    logs.create_log(loss, accuracy=accuracy, final_test=True, final_loss=result_loss)


if __name__ == "__main__":
    if hydra_available:
        for idx, s in enumerate(sys.argv):
            if "local_rank" in s:
                sys.argv[idx] = sys.argv[idx][2:]  # strip --
        hydra_main()
    else:
        args = arg_parser.parse_args()
        main(args)

