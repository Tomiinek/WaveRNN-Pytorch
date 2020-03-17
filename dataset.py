import numpy as np

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler, WeightedRandomSampler, SubsetRandomSampler

from hparams import hparams as hp
from utils import mulaw_quantize, inv_mulaw_quantize
import pickle
import csv
from audio import quantize


class RandomImbalancedSampler(Sampler):
    
    def __init__(self, data_source):
        lebel_freq = {}
        for idx in range(len(data_source)):
            label = data_source.items[idx]['language']
            if label in lebel_freq: lebel_freq[label] += 1
            else: lebel_freq[label] = 1

        total = float(sum(lebel_freq.values()))
        weights = [total / lebel_freq[data_source.items[idx]['language']] for idx in range(len(data_source))]
        self._sampler = WeightedRandomSampler(weights, len(weights))

    def __iter__(self):
        return self._sampler.__iter__()

    def __len__(self):
        return len(self._sampler)


class TextToSpeechDataset(Dataset):
    def __init__(self, data_path):
        self.path = os.path.join(data_path, "")
        with open(os.path.join(self.path, 'dataset.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            self.items = [{'id' : x[0], 'language' : x[1]} for x in metadata]
        self.mel_path = os.path.join(data_path, "mel")
        self.quant_path = os.path.join(data_path, "quant")
        self.test_path = os.path.join(data_path, "test")

    def __getitem__(self, index):
        f_id = self.items[index]['id']
        m = np.load(os.path.join(self.mel_path, f'{f_id}.npy'))
        x = np.load(os.path.join(self.quant_path, f'{f_id}.npy'))
        return m, x

    def __len__(self):
        return len(self.items)


class AudiobookDataset(Dataset):
    def __init__(self, data_path):
        self.path = os.path.join(data_path, "")
        with open(os.path.join(self.path,'dataset_ids.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)
        self.mel_path = os.path.join(data_path, "mel")
        self.wav_path = os.path.join(data_path, "wav")
        self.test_path = os.path.join(data_path, "test")
        
    def __getitem__(self, index):
        file = self.metadata[index]
        m = np.load(os.path.join(self.mel_path,'{}.npy'.format(file)))
        x = np.load(os.path.join(self.wav_path,'{}.npy'.format(file)))
        return m, x

    def __len__(self):
        return len(self.metadata)


class TacotronDataset(Dataset):
    def __init__(self, data_path):
        self.metadata=[]
        self.path = os.path.join(data_path, "")
        with open(os.path.join(self.path,'train.txt'), 'r', newline='') as f:
            csvreader = csv.reader(f, delimiter='|')
            for row in csvreader:
                self.metadata.append(row)

        self.mel_path = os.path.join(data_path, "mels")
        self.wav_path = os.path.join(data_path, "audio")
        self.test_path = os.path.join(data_path, "mels")

    def __getitem__(self, index):
        entry = self.metadata[index]
        m = np.load(os.path.join(self.mel_path, entry[1])).T
        wav = np.load(os.path.join(self.wav_path, entry[0]))

        if hp.input_type == 'raw' or hp.input_type=='mixture':
            wav = wav.astype(np.float32)
        elif hp.input_type == 'mulaw':
            wav = mulaw_quantize(wav, hp.mulaw_quantize_channels).astype(np.int)
        elif hp.input_type == 'bits':
            wav = quantize(wav).astype(np.int)
        else:
            raise ValueError("hp.input_type {} not recognized".format(hp.input_type))
        return m, wav

    def __len__(self):
        return len(self.metadata)


class Tacotron2Dataset(Dataset):
    def __init__(self, data_path):
        self.metadata=[]
        self.path = os.path.join(data_path, "")
        with open(os.path.join(self.path,'train.txt'), 'r', newline='') as f:
            csvreader = csv.reader(f, delimiter='|')
            for row in csvreader:
                self.metadata.append(row)
        
        self.mel_path = os.path.join(data_path, "mels")
        self.wav_path = os.path.join(data_path, "audio")
        self.test_path = os.path.join(data_path, "mels")
    
    def __getitem__(self, index):
        entry = self.metadata[index]
        m = np.load(os.path.join(self.mel_path, entry[1])).T
        wav = np.load(os.path.join(self.wav_path, entry[0]))
        
        if hp.input_type == 'raw' or hp.input_type=='mixture':
            wav = wav.astype(np.float32)
        elif hp.input_type == 'mulaw':
            wav = mulaw_quantize(wav, hp.mulaw_quantize_channels).astype(np.int)
        elif hp.input_type == 'bits':
            wav = quantize(wav).astype(np.int)
        else:
            raise ValueError("hp.input_type {} not recognized".format(hp.input_type))
        return m, wav
    
    def __len__(self):
        return len(self.metadata)


class MozillaTTS(Dataset):
    def __init__(self, data_path):
        self.metadata=[]
        self.path = os.path.join(data_path, "")
        with open(os.path.join(self.path,'tts_metadata.csv'), 'r', newline='') as f:
            csvreader = csv.reader(f, delimiter='|')
            for row in csvreader:
                self.metadata.append(row)

        self.mel_path = os.path.join(data_path, "mel")
        self.wav_path = os.path.join(data_path, "audio")
        self.test_path = os.path.join(data_path, "mel")

    def __getitem__(self, index):
        entry = self.metadata[index]
        m = np.load(entry[2].strip())
        wav = np.load(entry[1].strip())

        if hp.input_type == 'raw' or hp.input_type=='mixture':
            wav = wav.astype(np.float32)
        elif hp.input_type == 'mulaw':
            wav = mulaw_quantize(wav, hp.mulaw_quantize_channels).astype(np.int)
        elif hp.input_type == 'bits':
            wav = quantize(wav).astype(np.int)
        else:
            raise ValueError("hp.input_type {} not recognized".format(hp.input_type))
        return m, wav

    def __len__(self):
        return len(self.metadata)



def raw_collate(batch) :
    """collate function used for raw wav forms, such as using beta/guassian/mixture of logistic
    """
    
    pad = 2
    mel_win = hp.seq_len // hp.hop_size + 2 * pad
    max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * hp.hop_size for offset in mel_offsets]
    
    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] \
            for i, x in enumerate(batch)]
    
    coarse = [x[1][sig_offsets[i]:sig_offsets[i] + hp.seq_len + 1] \
              for i, x in enumerate(batch)]
    
    mels = np.stack(mels).astype(np.float32)
    coarse = np.stack(coarse).astype(np.float32)
    
    mels = torch.FloatTensor(mels)
    coarse = torch.FloatTensor(coarse)
    
    x_input = coarse[:,:hp.seq_len]
    
    y_coarse = coarse[:, 1:]
    
    return x_input, mels, y_coarse



def discrete_collate(batch) :
    """collate function used for discrete wav output, such as 9-bit, mulaw-discrete, etc.
    """
    
    pad = 2
    mel_win = hp.seq_len // hp.hop_size + 2 * pad
    max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * hp.hop_size for offset in mel_offsets]
    
    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] \
            for i, x in enumerate(batch)]
    
    coarse = [x[1][sig_offsets[i]:sig_offsets[i] + hp.seq_len + 1] \
              for i, x in enumerate(batch)]
    
    mels = np.stack(mels).astype(np.float32)
    coarse = np.stack(coarse).astype(np.int64)
    
    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)
    if hp.input_type == 'bits':
        x_input = 2 * coarse[:, :hp.seq_len].float() / (2**hp.bits - 1.) - 1.
    elif hp.input_type == 'mulaw':
        x_input = inv_mulaw_quantize(coarse[:, :hp.seq_len], hp.mulaw_quantize_channels)
    
    y_coarse = coarse[:, 1:]
    
    return x_input, mels, y_coarse


def no_test_raw_collate():
    import matplotlib.pyplot as plt
    from test_utils import plot, plot_spec
    data_id_path = "data_dir/"
    data_path = "data_dir/"
    print(hp.seq_len)
    
    with open('{}dataset_ids.pkl'.format(data_id_path), 'rb') as f:
        dataset_ids = pickle.load(f)
    dataset = AudiobookDataset(data_path)
    print(len(dataset))

    data_loader = DataLoader(dataset, collate_fn=raw_collate, batch_size=32, 
                         num_workers=0, shuffle=True)

    x, m, y = next(iter(data_loader))
    print(x.shape, m.shape, y.shape)
    plot(x.numpy()[0]) 
    plot(y.numpy()[0])
    plot_spec(m.numpy()[0])


def test_discrete_collate():
    import matplotlib.pyplot as plt
    from test_utils import plot, plot_spec
    data_id_path = "data_dir/"
    data_path = "data_dir/"
    print(hp.seq_len)
    
    with open('{}dataset_ids.pkl'.format(data_id_path), 'rb') as f:
        dataset_ids = pickle.load(f)
    dataset = AudiobookDataset(data_path)
    print(len(dataset))

    data_loader = DataLoader(dataset, collate_fn=discrete_collate, batch_size=32, 
                         num_workers=0, shuffle=True)

    x, m, y = next(iter(data_loader))
    print(x.shape, m.shape, y.shape)
    plot(x.numpy()[0]) 
    plot(y.numpy()[0])
    plot_spec(m.numpy()[0])



def no_test_dataset():
    data_id_path = "data_dir/"
    data_path = "data_dir/"
    print(hp.seq_len)
    dataset = AudiobookDataset(data_path)
