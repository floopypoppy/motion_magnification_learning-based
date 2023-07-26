"""training the magnet
"""
import os
# import time
# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import trange
# from functools import partial
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

from config import Config
from magnet import MagNetTrain
from data import get_gen_ABC
from callbacks import gen_state_dict
from losses import criterion_mag

from torch.utils.tensorboard import SummaryWriter

def train(tune_config):
    # Configurations
    config = Config()
    cudnn.benchmark = True

    magnet = MagNetTrain().cuda()
    if config.pretrained_weights:
        magnet.load_state_dict(gen_state_dict(config.pretrained_weights))
    if torch.cuda.device_count() > 1:
        magnet = nn.DataParallel(magnet)
    criterion = nn.L1Loss().cuda()

    optimizer = optim.Adam(magnet.parameters(), lr=tune_config['lr'], betas=config.betas)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=1, min_lr=config.lr/0.5**4, patience=5)

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    print('Save_dir:', config.save_dir)

    # Data generator
    data_loader = get_gen_ABC(config, mode='train')
    # print('Number of training image pairs:', data_loader.data_len)

    writer = SummaryWriter(config.log_dir)

    # Training
    n_batch = 0
    # running_loss = 0.0
    # batch_to_eval = 1000

    for epoch in range(1, config.epochs+1):
        print('epoch:', epoch)
        # losses, losses_y, losses_texture_AC, losses_texture_BM, losses_motion_BC = [], [], [], [], []
        for _ in trange(0, data_loader.data_len, data_loader.batch_size):

            # Data Loading
            batch_A, batch_B, batch_C, batch_M, batch_amp = data_loader.gen()
            n_batch += 1

            # G Train
            optimizer.zero_grad()
            y_hat, texture_AC, texture_BM, motion_BC = magnet(batch_A, batch_B, batch_C, batch_M, batch_amp)
            loss_y, loss_texture_AC, loss_texture_BM, loss_motion_BC = criterion_mag(y_hat, batch_M, texture_AC, texture_BM, motion_BC, criterion)
    #         loss = loss_y + (loss_texture_AC + loss_texture_BM + loss_motion_BC) * 0.1
            loss = loss_y + (loss_texture_AC + loss_motion_BC) * 0.1
            loss.backward()
            optimizer.step()

            writer.add_scalar('Training loss by minibatch', loss, n_batch)

    #         learning rate scheduling
    #         running_loss += loss.item()
    #         if n_batch % batch_to_eval == 0:
    #             avg_loss = running_loss / batch_to_eval
    #             scheduler.step(avg_loss)
    #             running_loss = 0.0

    #             writer.add_scalar('Learning rate every 1000 minibatches',
    #                             optimizer.state_dict()['param_groups'][0]['lr'],
    #                             n_batch/batch_to_eval)

            # Callbacks
    #         losses.append(loss.item())
    #         losses_y.append(loss_y.item())
    #         losses_texture_AC.append(loss_texture_AC.item())
    # #         losses_texture_BM.append(loss_texture_BM.item())
    #         losses_motion_BC.append(loss_motion_BC.item())

        # Collections
        # save_model(magnet.state_dict(), losses, config.save_dir, epoch)

        # print('\ntime: {}m, ep: {} \nloss: {:.3e}, y: {:.3e}, tex_AC: {:.3e}, mot_BC: {:.3e}\n'.format(int((time.time()-config.time_st)/60), epoch, np.mean(losses), np.mean(losses_y), np.mean(losses_texture_AC), np.mean(losses_motion_BC) ))

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for _ in trange(0, data_loader.validate_data_len, data_loader.batch_size):
            with torch.no_grad():
                batch_A, batch_B, batch_C, batch_M, batch_amp = data_loader.gen_validate()

                y_hat, texture_AC, texture_BM, motion_BC = magnet(batch_A, batch_B, batch_C, batch_M, batch_amp)
                loss_y, loss_texture_AC, loss_texture_BM, loss_motion_BC = criterion_mag(y_hat, batch_M, texture_AC,
                                                                                         texture_BM, motion_BC, criterion)
                loss = loss_y + (loss_texture_AC + loss_motion_BC) * 0.1

                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": magnet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_loss / val_steps},
            checkpoint=checkpoint,
        )

def main(num_samples=10, max_num_epochs=12, gpus_per_trial=1):

    tune_config = {
        "lr": tune.loguniform(1e-5, 1e-3),
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        train,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=tune_config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    # best_trained_model.to(device)

    # best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    # best_checkpoint_data = best_checkpoint.to_dict()

    # best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=12, gpus_per_trial=1)