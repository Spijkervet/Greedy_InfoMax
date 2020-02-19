import os
import argparse
import torch
import time
import numpy as np
from apex import amp

from torch.utils.tensorboard import SummaryWriter

try:
    from sacred import Experiment
    from sacred.stflow import LogFileWriter
    from sacred.observers import FileStorageObserver, MongoObserver

    sacred_available = True

    #### pass configuration
    ex = Experiment("experiment")

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
from GreedyInfoMax.vision.arg_parser import arg_parser
from GreedyInfoMax.vision.models import load_vision_model
from GreedyInfoMax.vision.data import get_dataloader


def validate(opt, model, test_loader):
    total_step = len(test_loader)

    loss_epoch = [0 for i in range(opt.model_splits)]
    starttime = time.time()

    for step, (img, label) in enumerate(test_loader):

        model_input = img.to(opt.device)
        label = label.to(opt.device)

        loss, _, _, _ = model(model_input, label, n=opt.train_module)
        loss = torch.mean(loss, 0)

        loss_epoch += loss.data.cpu().numpy()

    for i in range(opt.model_splits):
        print(
            "Validation Loss Model {}: Time (s): {:.1f} --- {:.4f}".format(
                i, time.time() - starttime, loss_epoch[i] / total_step
            )
        )

    validation_loss = [x / total_step for x in loss_epoch]
    return validation_loss


def train(opt, model, optimizer, writer, logs):

    (
        train_loader,
        _,
        supervised_loader,
        _,
        test_loader,
        _,
    ) = get_dataloader.get_dataloader(opt)

    if opt.loss == 1:
        train_loader = supervised_loader

    total_step = len(train_loader)
    model.module.switch_calc_loss(True)

    print_idx = 1

    starttime = time.time()
    cur_train_module = opt.train_module

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):

        loss_epoch = [0 for i in range(opt.model_splits)]
        loss_updates = [1 for i in range(opt.model_splits)]

        for step, (img, label) in enumerate(train_loader):

            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Training Block: {}, Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs + opt.start_epoch,
                        step,
                        total_step,
                        cur_train_module,
                        time.time() - starttime,
                    )
                )

            starttime = time.time()

            model_input = img.to(opt.device)
            label = label.to(opt.device)

            loss, _, _, accuracy = model(model_input, label, n=cur_train_module)
            loss = torch.mean(loss, 0)  # take mean over outputs of different GPUs
            accuracy = torch.mean(accuracy, 0)

            if cur_train_module != opt.model_splits and opt.model_splits > 1:
                loss = loss[cur_train_module].unsqueeze(0)

            # loop through the losses of the modules and do gradient descent
            for idx, cur_losses in enumerate(loss):
                if len(loss) == 1 and opt.model_splits != 1:
                    idx = cur_train_module

                model.zero_grad()

                if opt.fp16:
                    with amp.scale_loss(cur_losses, optimizer) as scaled_loss:
                        if idx == len(loss) - 1:
                            scaled_loss.backward()
                        else:
                            scaled_loss.backward(retain_graph=True)
                else:
                    if idx == len(loss) - 1:
                        cur_losses.backward()
                    else:
                        cur_losses.backward(retain_graph=True)

                optimizer[idx].step()

                print_loss = cur_losses.item()
                print_acc = accuracy[idx].item()
                if step % print_idx == 0:
                    print("\t \t Loss: \t \t {:.4f}".format(print_loss))
                    if opt.loss == 1:
                        print("\t \t Accuracy: \t \t {:.4f}".format(print_acc))

                loss_epoch[idx] += print_loss
                loss_updates[idx] += 1

        if opt.validate:
            validation_loss = validate(
                opt, model, test_loader
            )  # test_loader corresponds to validation set here
            logs.append_val_loss(validation_loss)

        logs.append_train_loss(
            [x / loss_updates[idx] for idx, x in enumerate(loss_epoch)]
        )
        logs.create_log(model, epoch=epoch, optimizer=optimizer)


def main(opt, experiment_name):
    # set start time
    opt.time = time.ctime()

    # Device configuration
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    arg_parser.create_log_path(opt)
    opt.training_dataset = "unlabeled"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    model, optimizer = load_vision_model.load_model_and_optimizer(opt)

    logs = logger.Logger(opt)

    # set comment to experiment's name
    tb_dir = os.path.join(opt.log_path, experiment_name)
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    try:
        # Train the model
        train(opt, model, optimizer, writer, logs)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(model)


if sacred_available:
    from GreedyInfoMax.utils.yaml_config_hook import yaml_config_hook

    @ex.config
    def my_config():
        yaml_config_hook("./config/vision/config.yaml", ex)

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
        opt = arg_parser.parse_args()
        opt.use_sacred = False
        print(opt)
        main(opt, experiment_name="greedy_infomax")

