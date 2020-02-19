import sys
import argparse
import torch
import time
import os
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
from GreedyInfoMax.audio.data import get_dataloader, phone_dict
from GreedyInfoMax.utils import logger, utils
from GreedyInfoMax.audio.arg_parser import arg_parser
from GreedyInfoMax.audio.models import load_audio_model


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)


def train(
    opt,
    train_dataset,
    phone_dict,
    context_model,
    model,
    optimizer,
    n_features,
    writer,
    logs
):
    total_step = len(train_dataset.file_list)
    criterion = torch.nn.CrossEntropyLoss()
    total_i = 0
    for epoch in range(opt.num_epochs):
        loss_epoch = 0
        acc_epoch = 0

        for i, k in enumerate(train_dataset.file_list):
            starttime = time.time()

            audio, filename = train_dataset.get_full_size_test_item(i)

            ### get latent representations for current audio
            model_input = audio.to(opt.device)
            model_input = torch.unsqueeze(model_input, 0)

            targets = torch.LongTensor(phone_dict[filename])
            targets = targets.to(opt.device).reshape(-1)

            if opt.model_type == 2:  ##fully supervised training
                for idx, layer in enumerate(context_model.module.fullmodel):
                    context, z = layer.get_latents(model_input)
                    model_input = z.permute(0, 2, 1)
            else:
                with torch.no_grad():
                    for idx, layer in enumerate(context_model.module.fullmodel):
                        if idx + 1 < len(context_model.module.fullmodel):
                            _, z = layer.get_latents(
                                model_input, calc_autoregressive=False
                            )
                            model_input = z.permute(0, 2, 1)
                    context, _ = context_model.module.fullmodel[idx].get_latents(
                        model_input, calc_autoregressive=True
                    )
                context = context.detach()

            inputs = context.reshape(-1, n_features)

            # forward pass
            output = model(inputs)

            """ 
            The provided phone labels are slightly shorter than expected, 
            so we cut our predictions to the right length.
            Cutting from the front gave better results empirically.
            """
            output = output[-targets.size(0) :]  # output[ :targets.size(0)]

            loss = criterion(output, targets)

            # calculate accuracy
            (accuracy,) = utils.accuracy(output.data, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sample_loss = loss.item()
            loss_epoch += sample_loss
            acc_epoch += accuracy

            writer.add_scalar("Loss/train_step", sample_loss, total_i)
            writer.add_scalar("Accuracy/train_step", accuracy, total_i)
            writer.flush()

            if i % 10 == 0:
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
            total_i += 1

        logs.append_train_loss([loss_epoch / total_step])
        logs.create_log(model, epoch=epoch, accuracy=accuracy)
        writer.add_scalar("Loss/train_epoch", loss_epoch / total_step, epoch)
        writer.add_scalar("Accuracy/train_epoch", acc_epoch / total_step, epoch)
        writer.flush()
        
        # Sacred
        ex.log_scalar("train.loss", loss_epoch / total_step)
        ex.log_scalar("train.accuracy", acc_epoch / total_step)


def test(
    opt, test_dataset, phone_dict, context_model, model, optimizer, n_features, logs
):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for idx, k in enumerate(test_dataset.file_list):

            audio, filename = test_dataset.get_full_size_test_item(idx)

            model.zero_grad()

            ### get latent representations for current audio
            model_input = audio.to(opt.device)
            model_input = torch.unsqueeze(model_input, 0)

            targets = torch.LongTensor(phone_dict[filename])

            with torch.no_grad():
                for idx, layer in enumerate(context_model.module.fullmodel):
                    if idx + 1 < len(context_model.module.fullmodel):
                        _, z = layer.get_latents(model_input, calc_autoregressive=False)
                        model_input = z.permute(0, 2, 1)
                context, _ = context_model.module.fullmodel[idx].get_latents(
                    model_input, calc_autoregressive=True
                )

                context = context.detach()

                targets = targets.to(opt.device).reshape(-1)
                inputs = context.reshape(-1, n_features)

                # forward pass
                output = model(inputs)

            output = output[-targets.size(0) :]

            # calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if idx % 1000 == 0:
                print(
                    "Step [{}/{}], Accuracy: {:.4f}".format(
                        idx, len(test_dataset.file_list), correct / total
                    )
                )

    accuracy = (correct / total) * 100
    print("Final Testing Accuracy: ", accuracy)
    return accuracy


def main(opt, experiment_name):
    opt.time = time.ctime()

    # Device configuration
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt.batch_size = 8
    opt.num_epochs = 20

    arg_parser.create_log_path(opt, add_path_var="linear_model")

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load self-supervised GIM model
    context_model, _ = load_audio_model.load_model_and_optimizer(opt, reload_model=True)

    if opt.model_type != 2:  # == 2 trains a fully supervised model
        context_model.eval()

    # 41 different phones to differentiate
    n_classes = 41
    n_features = context_model.module.reg_hidden

    # create linear classifier
    model = torch.nn.Sequential(torch.nn.Linear(n_features, n_classes)).to(opt.device)
    model.apply(weights_init)

    if opt.model_type == 2:
        params = list(context_model.parameters()) + list(model.parameters())
    else:
        params = model.parameters()

    optimizer = torch.optim.Adam(params, lr=1e-4)

    # load dataset
    pd = phone_dict.load_phone_dict(opt)
    _, train_dataset, _, test_dataset = get_dataloader.get_libri_dataloaders(opt)

    logs = logger.Logger(opt)
    accuracy = 0

    # set comment to experiment's name
    tb_dir = os.path.join(opt.log_path, experiment_name)
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    try:
        # Train the model
        train(opt, train_dataset, pd, context_model, model, optimizer, n_features, writer, logs)

        # Test the model
        accuracy = test(
            opt, test_dataset, pd, context_model, model, optimizer, n_features, logs
        )

    except KeyboardInterrupt:
        print("Training interrupted, saving log files")

    logs.create_log(model, accuracy=accuracy, final_test=True)

    if opt.model_type == 2:
        print("Saving supervised model")
        torch.save(
            context_model.state_dict(), os.path.join(opt.log_path, "context_model.ckpt")
        )


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
        main(args, experiment_name="logistic_regression_phones")
