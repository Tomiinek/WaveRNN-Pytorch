import os
import librosa
import glob

from docopt import docopt
from model import *
from hparams import hparams
from audio import normalize
import pickle
import time
import numpy as np
import scipy as sp


if __name__ == "__main__":
    import argparse
    import os
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", type=str, default=".", help="Base directory of the project.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Name of the initial checkpoint.")
    parser.add_argument('--hyper_parameters', type=str, default=None, help="Name of the hyperparameters file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input spectrogram.")
    parser.add_argument("--output", type=str, default=".", help="Path to output directory.")
    parser.add_argument("--no_cuda", action='store_true', help="Force to run on CPU.")
    args = parser.parse_args()
    
    device = torch.device("cpu" if args.no_cuda else "cuda")

    if args.hyper_parameters is not None:
        with open(args.hyper_parameters) as f:
            hparams.parse_json(f.read())

    mel = np.load(args.input)
    if mel.shape[0] > mel.shape[1]: #ugly hack for transposed mels
        mel = mel.T
    mel = normalize()
    
    model = build_model().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.train(False)

    mel0 = mel.copy()
    mel0 = np.hstack([np.ones([80,40])*(-4), mel0, np.ones([80,40])*(-4)])
    start = time.time()
    output0 = model.generate(mel0, batched=False, target=2000, overlap=64)
    total_time = time.time() - start
    frag_time = len(output0) / hparams.sample_rate
    print(f"Generation time: {total_time}. Sound time: {frag_time}, ratio: {frag_time/total_time}")

    librosa.output.write_wav(os.path.join(args.output, os.path.basename(args.input) + '_orig.wav'), output0, hparams.sample_rate)
