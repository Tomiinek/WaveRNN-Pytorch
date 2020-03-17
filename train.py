"""Training WaveRNN Model.

usage: train.py [options] <data-root>

options:
    --checkpoint-dir=<dir>      Directory where to save model checkpoints [default: checkpoints].
    --checkpoint=<path>         Restore model from checkpoint path if given.
    --log-event-path=<path>     Path to tensorboard event log
    --dataset=<name>            Dataset type Tacotron, TTS, Audiobooks [default: Tacotron].
    -h, --help                  Show this help message and exit
"""
import os

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from docopt import docopt
from os.path import join
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import raw_collate, discrete_collate, AudiobookDataset, TacotronDataset, MozillaTTS, TextToSpeechDataset, RandomImbalancedSampler
from distributions import *
from hparams import hparams as hp
from loss_function import nll_loss
from lrschedule import noam_learning_rate_decay, step_learning_rate_decay
from model import build_model
from utils import num_params_count

global_step = 0
global_epoch = 0
global_test_step = 0
use_cuda = torch.cuda.is_available()


def np_now(tensor):
    return tensor.detach().cpu().numpy()


def clamp(x, lo=0, hi=1):
    return max(lo, min(hi, x))


class PruneMask():
    def __init__(self, layer, prune_rnn_input):
        self.mask = []
        self.p_idx = [0]
        self.total_params = 0
        self.pruned_params = 0
        self.split_size = 0
        self.init_mask(layer, prune_rnn_input)

    def init_mask(self, layer, prune_rnn_input):
        # Determine the layer type and
        # num matrix splits if rnn
        layer_type = str(layer).split('(')[0]
        splits = {'Linear': 1, 'GRU': 3, 'LSTM': 4}

        # Organise the num and indices of layer parameters
        # Dense will have one index and rnns two (if pruning input)
        if layer_type != 'Linear':
            self.p_idx = [0, 1] if prune_rnn_input else [1]

        # Get list of parameters from layers
        params = self.get_params(layer)

        # For each param matrix in this layer, create a mask
        for W in params:
            self.mask += [torch.ones_like(W)]
            self.total_params += W.size(0) * W.size(1)

        # Need a split size for mask_from_matrix() later on
        self.split_size = self.mask[0].size(0) // splits[layer_type]

    def get_params(self, layer):
        params = []
        for idx in self.p_idx:
            params += [list(layer.parameters())[idx].data]
        return params

    def update_mask(self, layer, z):
        params = self.get_params(layer)
        for i, W in enumerate(params):
            self.mask[i] = self.mask_from_matrix(W, z)
        self.update_prune_count()

    def apply_mask(self, layer):
        params = self.get_params(layer)
        for M, W in zip(self.mask, params):
            W *= M

    def mask_from_matrix(self, W, z):
        # Split into gate matrices (or not)
        if self.split_size>1:
            W_split = torch.split(W, self.split_size)
        else:
            W_split = W

        M = []
        # Loop through splits
        for W in W_split:
            # Sort the magnitudes
            N = W.shape[1]

            W_abs = torch.abs(W)
            L = W_abs.reshape(W.shape[0], N // hp.sparse_group, hp.sparse_group)
            S = L.sum(dim=2)
            sorted_abs, _ = torch.sort(S.view(-1))

            # Pick k (num weights to zero)
            k = int(W.shape[0] * W.shape[1] // hp.sparse_group * z)
            threshold = sorted_abs[k]
            mask = (S >= threshold).float()
            mask = mask.unsqueeze(2).expand(-1,-1,hp.sparse_group)
            mask = mask.reshape(W.shape[0], W.shape[1])

            # Create the mask
            M += [mask]

        return torch.cat(M)

    def update_prune_count(self):
        self.pruned_params = 0
        for M in self.mask:
            self.pruned_params += int(np_now((M - 1).sum() * -1))


class Pruner(object):
    def __init__(self, layers, start_prune, prune_steps, target_sparsity,
                 prune_rnn_input=True):
        self.z = 0  # Objects sparsity @ time t
        self.t_0 = start_prune
        self.S = prune_steps
        self.Z = target_sparsity
        self.num_pruned = 0
        self.total_params = 0
        self.masks = []
        self.layers = layers
        for (layer,z) in layers:
            self.masks += [PruneMask(layer, prune_rnn_input)]
        self.count_total_params()

    def update_sparsity(self, t, Z):
        z = Z * (1 - (1 - (t - self.t_0) / self.S) ** 3)
        z = clamp(z, 0, Z)
        return z

    def prune(self, step):
        for ((l,z), m) in zip(self.layers, self.masks):
            z_curr = self.update_sparsity(step, z)
            m.update_mask(l, z_curr)
            m.apply_mask(l)
        return self.count_num_pruned(), z_curr

    def restart(self, layers, step):
        # In case training is stopped
        self.update_sparsity(step)
        for ((l, z), m) in zip(layers, self.masks):
            z_curr = self.update_sparsity(step, z)
            m.update_mask(l, z_curr)

    def count_num_pruned(self):
        self.num_pruned = 0
        for m in self.masks:
            self.num_pruned += m.pruned_params
        return self.num_pruned

    def count_total_params(self):
        for m in self.masks:
            self.total_params += m.total_params
        return self.total_params


def save_checkpoint(device, model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(step))
    optimizer_state = optimizer.state_dict()
    global global_test_step
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch
    global global_test_step

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as e:
                print(e)
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    global_test_step = checkpoint.get("global_test_step", 0)

    return model


def evaluate_model(model, data_loader, checkpoint_dir, limit_eval_to=5):
    """evaluate model and save generated wav and plot

    """
    test_path = data_loader.dataset.test_path
    test_files = os.listdir(test_path)
    counter = 0
    output_dir = os.path.join(checkpoint_dir, 'eval')
    for f in test_files:
        if (f[-7:] == "mel.npy") or ('mel' in f):
            mel = np.load(os.path.join(test_path, f))
            if mel.shape[-1]==hp.num_mels: #fix the order
                mel = mel.T
            wav = model.generate(mel, batched=True)
            # save wav
            wav_path = os.path.join(output_dir, "checkpoint_step{:09d}_wav_{}.wav".format(global_step, counter))
            librosa.output.write_wav(wav_path, wav.astype('float32'), sr=hp.sample_rate)
            # save wav plot
            fig_path = os.path.join(output_dir, "checkpoint_step{:09d}_wav_{}.png".format(global_step, counter))
            fig = plt.plot(wav.reshape(-1))
            plt.savefig(fig_path)
            # clear fig to drawing to the same plot
            plt.clf()

            if counter == 0:
                wav = model.generate(mel, batched=False)
                # save wav
                wav_path = os.path.join(output_dir,
                                        "checkpoint_step{:09d}_wav_unbatched_{}.wav".format(global_step, counter))
                librosa.output.write_wav(wav_path, wav.astype('float32'), sr=hp.sample_rate)
                # save wav plot
                fig_path = os.path.join(output_dir,
                                        "checkpoint_step{:09d}_wav_unbatched_{}.png".format(global_step, counter))
                fig = plt.plot(wav.reshape(-1))
                plt.savefig(fig_path)
                # clear fig to drawing to the same plot
                plt.clf()

            counter += 1

        # stop evaluation early via limit_eval_to
        if counter >= limit_eval_to:
            break


def train_loop(device, model, data_loader, optimizer, checkpoint_dir):
    
    # create loss and put on device
    if hp.input_type == 'raw':
        if hp.distribution == 'beta':
            criterion = beta_mle_loss
        elif hp.distribution == 'gaussian':
            criterion = gaussian_loss
    elif hp.input_type == 'mixture':
        criterion = discretized_mix_logistic_loss
    elif hp.input_type in ["bits", "mulaw"]:
        criterion = nll_loss
    else:
        raise ValueError("input_type:{} not supported".format(hp.input_type))

    # Pruner for reducing memory footprint
    layers = [(model.I,hp.sparsity_target), (model.rnn1,hp.sparsity_target_rnn), (model.fc1,hp.sparsity_target), (model.fc3,hp.sparsity_target)] #(model.fc2,hp.sparsity_target),
    pruner = Pruner(layers, hp.start_prune, hp.prune_steps, hp.sparsity_target)

    global global_step, global_epoch, global_test_step
    while global_epoch < hp.nepochs:
        running_loss = 0
        for i, (x, m, y) in enumerate(tqdm(data_loader)):
            x, m, y = x.to(device), m.to(device), y.to(device)
            y_hat = model(x, m)
            y = y.unsqueeze(-1)
            loss = criterion(y_hat, y)
            # calculate learning rate and update learning rate
            if hp.fix_learning_rate:
                current_lr = hp.fix_learning_rate
            elif hp.lr_schedule_type == 'step':
                current_lr = step_learning_rate_decay(hp.initial_learning_rate, global_step, hp.step_gamma,
                                                      hp.lr_step_interval)
            else:
                current_lr = noam_learning_rate_decay(hp.initial_learning_rate, global_step, hp.noam_warm_up_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), hp.grad_norm)
            optimizer.step()
            num_pruned, z = pruner.prune(global_step)

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)

            writer.add_scalar("loss", float(loss.item()), global_step)
            writer.add_scalar("avg_loss", float(avg_loss), global_step)
            writer.add_scalar("learning_rate", float(current_lr), global_step)
            writer.add_scalar("grad_norm", float(grad_norm), global_step)
            writer.add_scalar("num_pruned", float(num_pruned), global_step)
            writer.add_scalar("fraction_pruned", z, global_step)

            # saving checkpoint if needed
            if global_step != 0 and global_step % hp.save_every_step == 0:
                pruner.prune(global_step)
                save_checkpoint(device, model, optimizer, global_step, checkpoint_dir, global_epoch)
            # evaluate model if needed
            if global_step != 0 and global_test_step != True and global_step % hp.evaluate_every_step == 0:
                pruner.prune(global_step)
                print("step {}, evaluating model: generating wav from mel...".format(global_step))
                evaluate_model(model, data_loader, checkpoint_dir)
                print("evaluation finished, resuming training...")

            # reset global_test_step status after evaluation
            if global_test_step is True:
                global_test_step = False
            global_step += 1

        print("epoch:{}, running loss:{}, average loss:{}, current lr:{}, num_pruned:{} ({}%)".format(global_epoch, running_loss, avg_loss, current_lr, num_pruned, z))
        global_epoch += 1


if __name__ == "__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", type=str, default=".", help="Base directory of the project.")
    parser.add_argument("--data_root", type=str, default="dataset", help="Base of input directories.")
    parser.add_argument("--checkpoint_directory", type=str, default="checkpoints", help="Checkpoints directory.")
    parser.add_argument("--log_directory", type=str, default="logs", help="Logs directory.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint name.")
    
    args = parser.parse_args()

    checkpoint_dir = os.path.join(args.base_directory, args.checkpoint_directory)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.join(checkpoint_dir, "eval")):
        os.makedirs(os.path.join(checkpoint_dir, "eval"))

    log_dir = os.path.join(args.base_directory, args.log_directory)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    dataset = TextToSpeechDataset(args.data_root)

    if hp.input_type == 'raw':
        collate_fn = raw_collate
    elif hp.input_type == 'mixture':
        collate_fn = raw_collate
    elif hp.input_type in ['bits', 'mulaw']:
        collate_fn = discrete_collate

    sampler = RandomImbalancedSampler(dataset)
    data_loader = DataLoader(dataset, collate_fn=collate_fn, shuffle=False, 
                             sampler=sampler, num_workers=int(hp.num_workers), batch_size=hp.batch_size)
    device = torch.device("cuda" if use_cuda else "cpu")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_dir = os.path.join(args.base_directory, args.log_directory, f'log-{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)

    # build model, create optimizer
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=hp.initial_learning_rate, betas=(hp.adam_beta1, hp.adam_beta2),
                           eps=hp.adam_eps, weight_decay=hp.weight_decay, amsgrad=hp.amsgrad)

    # load checkpoint
    if args.checkpoint is not None:
        model = load_checkpoint(args.checkpoint, model, optimizer, True)
        # set global_test_step to True so we don't evaluate right when we load in the model
        global_test_step = True

    # main train loop
    try:
        train_loop(device, model, data_loader, optimizer, args.checkpoint_directory)
    except KeyboardInterrupt:
        print("Interrupted!")
        pass
    except Exception as e:
        print(e)
    finally:
        save_checkpoint(device, model, optimizer, global_step, args.checkpoint_directory, global_epoch)
