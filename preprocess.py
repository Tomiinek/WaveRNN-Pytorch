import os
from docopt import docopt
import numpy as np
import math, pickle, os
from audio import *
from hparams import hparams as hp
from utils import *
from tqdm import tqdm


def get_wav_mel(path, name):
    """
    Given path and name of a .wav file, get the quantized wav and mel spectrogram as numpy vectors.
    """
    wav = load_wav(os.path.join(path, "wavs", name + ".wav"))
    mel = np.load(os.path.join(path, "gtas", name + ".npy"))
    mel = normalize(mel)
    if hp.input_type == 'raw' or hp.input_type=='mixture':
        return wav.astype(np.float32), mel
    elif hp.input_type == 'mulaw':
        quant = mulaw_quantize(wav, hp.mulaw_quantize_channels)
        return quant.astype(np.int), mel
    elif hp.input_type == 'bits':
        quant = quantize(wav)
        return quant.astype(np.int), mel
    else:
        raise ValueError(f"hp.input_type {hp.input_type} not recognized")


def process_data(data_root, data_dirs, output_path, num_test_per_dir=4):
    """
    Given language dependent directories and an output directory, 
    process wav files and save quantized wav and mel.
    """

    dataset_info = []
    file_names = []
    test_file_names = []
    
    c = 0
    for d in data_dirs:
        wav_d = os.path.join(data_root, d, "wavs")
        all_files = [os.path.splitext(f)[0] for f in os.listdir(wav_d)]
        
        for i, f in enumerate(all_files):
            c += 1
            file_id = '{:d}'.format(c).zfill(5)
            wav, mel = get_wav_mel(os.path.join(data_root, d), f)
            if i < num_test_per_dir:
                np.save(os.path.join(output_path, "test", f"test_{d}_{c}_mel.npy"), mel)
                np.save(os.path.join(output_path, "test", f"test_{d}_{c}_wav.npy"), wav) 
            else:
                np.save(os.path.join(output_path, "mel",   file_id + ".npy"), mel)
                np.save(os.path.join(output_path, "quant", file_id + ".npy"), wav) 
                dataset_info.append((file_id, os.path.basename(d)))

    # save dataset
    with open(os.path.join(output_path, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset_info, f)
    
    print(f"Preprocessing done, total processed wav files: {len(wav_files)}")
    print(f"Processed files are located in:{os.path.abspath(output_path)}")


if __name__=="__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", type=str, default=".", help="Base directory of the project.")
    parser.add_argument("--output", type=str, default="output", help="Output directory.", required=True)
    parser.add_argument("--data_root", type=str, default="data", help="Base of input directories.")
    parser.add_argument("--inputs", nargs='+', type=str, help="Names of input directories.", required=True)
    args = parser.parse_args()

    output_dir = os.path.join(args.base_directory, args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_quant_dir = os.path.join(args.base_directory, args.output, "quant")
    if not os.path.exists(output_quant_dir):
        os.makedirs(output_quant_dir)

    output_mel_dir = os.path.join(args.base_directory, args.output, "mel")
    if not os.path.exists(output_mel_dir):
        os.makedirs(output_mel_dir)

    output_test_dir = os.path.join(args.base_directory, args.output, "test")
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)

    # process data
    process_data(args.data_root, args.inputs, output_dir)
