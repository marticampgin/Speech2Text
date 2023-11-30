from typing import List, Any
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

import os
import torch
import librosa 
import numpy as np
import skimage.io
import pandas as pd
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class CustomSpectroDataset(Dataset):
    def __init__(self,
                 csv_data : str,
                 recordings_folder : str, 
                 spectr_folder : str,
                 produce_spectr : bool, 
                 transform : Any=None,
                 ):
        
        self.transform = transform
        self.df = pd.read_csv(csv_data)
        self.df.dropna(how='any', inplace=True)
        self.spectr_folder = spectr_folder
        self.jpg_paths = os.listdir(self.spectr_folder)
        self.wav_filenames = self.df['channel_1'].tolist()
        self.transcriptions = self.df['transcription'].tolist()
        assert len(self.wav_filenames) == len(self.transcriptions)

        # If we want to spectrograms from WAV-files (not needed if images are already present)
        if produce_spectr:
            for name in tqdm(self.wav_filenames):
                if os.path.exists(os.path.join(recordings_folder, name)):
                    self._logar(os.path.join(recordings_folder, name), spectr_folder)
            print('Conversion done! Now launch the program again with --wav2spec argument set to False.')
            exit()
        self.data = []

        for i, wav in enumerate(tqdm(self.wav_filenames)):
            for jpg in self.jpg_paths:
                if jpg.split('.')[0] == wav.split('.')[0]:
                    self.data.append({'img_name': jpg, 'seq': self.transcriptions[i], 'wav': wav})
        
        
        self.int2str = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]'}
        
        all_tokens = []
        
        # SIMPLE TOKENIZER: SPLIT ON SPACEBAR
        for sample in self.data:
            transcription = sample['seq']
            tokens = transcription.split()
            all_tokens.extend(tokens)

        vocab = list(set(all_tokens))

        for token in vocab:
            self.int2str[len(self.int2str)] = token
        
        self.str2int = {st:i for i, st in self.int2str.items()}

    def __getitem__(self, index : int) -> dict[str, torch.tensor]:
        img_name = self.data[index]['img_name']
        seq = self.data[index]['seq']
        wav = self.data[index]['wav']

        with Image.open(os.path.join(self.spectr_folder, img_name)) as img_file:
            if self.transform:
                img_tensor = self.transform(img_file)

        # seq_to_int = [self.tokenizer.vocab['[BOS]']]
        seq_to_int = [self.str2int['[BOS]']]
        seq_to_int.extend([self.str2int[token] for token in seq.split()])
        seq_to_int.append(self.str2int['[EOS]'])

        return {'img': img_tensor, 
                'seq': torch.tensor(seq_to_int),
                'wav': wav}

    def get_vocab_size(self) -> List[str]:
        #return len(self.tokenizer.vocab)
        return len(self.int2str)
    
    
    def get_all_transcriptions(self) -> List[dict[torch.tensor, str]]:
        transcriptions = [sample['seq'] for sample in self.data]
        return transcriptions


    def __len__(self) -> int:
        return len(self.data)
      

    def _logar(self, wav_file_path : str, jpg_folder_output : str) -> None:
        """
        Prdouces black and white spectrogram, log-scale Mel spectrogram. 
        """
        
        # Hardcoded parameters
        n_fft = 500 # window length
        hop_length = 200 # number of samples per time-step in spectrogram
        n_mels = 128 # number of bins in spectrogram
        sr = 22050 # sampling rate 

        y, sr = librosa.load(wav_file_path, sr=sr)
        
        file_name = os.path.splitext(os.path.basename(wav_file_path))[0]
        output_folder = jpg_folder_output
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{file_name}.jpg")

        self._spectrogram_image(y, sr=sr, out=output_path, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)


    def _scale_minmax(self, X, min=0.0, max=1.0):
        """
        Applies min-max scale to fit inside 8-bit range
        """
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled


    def _spectrogram_image(self, y, sr, out, hop_length, n_fft, n_mels):
        # use log-melspectrogram
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                                n_fft=n_fft, hop_length=hop_length)
        mels = np.log(mels + 1e-9) # add small number to avoid log(0)
        img = self._scale_minmax(mels, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0) # put low frequencies at the bottom in image
        img = 255-img # invert. make black==more energy

        # save as PNG
        skimage.io.imsave(out, img)


class CollateFunctor:
    """
    Simple collator to pad decoder sequences
    """

    def __init__(self, pad_idx : int) -> None:
        self.pad_idx = pad_idx
        self.max = 1024
        
    def __call__(self, samples) -> dict[str, torch.tensor]:
        img_tensors = []
        sequences = []
        wavs = []

        for sample in sorted(samples, key=lambda x: len(x['seq']), reverse=True):
            img_tensors.append(sample['img'])
            sequences.append(sample['seq'])
            wavs.append(sample['wav'])

        # Padding sequences
        padded_seq_tensors = pad_sequence(sequences, 
                                          batch_first=True,
                                          padding_value=self.pad_idx)
        
        # Padding images
        for i, tensor in enumerate(img_tensors):
            img_tensors[i] = F.pad(tensor, (0, self.max - tensor.shape[-1]), 'constant', 0)
        
        img_tensors = torch.stack(tuple(img_tensors), dim=0)
        return {'img': img_tensors,
                'seq': padded_seq_tensors,
                'wav': wavs}
