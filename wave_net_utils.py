
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import torchvision.transforms.functional as TF
from typing import List, Tuple, Dict, Set, Union

def load_voices_files(voice_files):
    voices = []
    for voice_file in voice_files:
        _, samples = wavfile.read(voice_file, True)
        voices.append(samples.tolist())
    return voices

def get_max_length(voices):
    max_length = 0
    for voice in voices:
        voice_length = len(voice)
        if voice_length > max_length:
            max_length = voice_length
    return max_length


def fill_voices_data_with_pads(voices):
    padded_voices = []
    lengths = []
    max_length = get_max_length(voices)
    for voice in voices:
        voice_length = len(voice)
        lengths.append(voice_length)
        padded_voices.append(voice+[0.0]*(max_length-voice_length))
    return padded_voices, lengths

def quantize_voices(voices: np.array, u: int):
    rs = np.array(voices)
    rs = np.sign(rs)*np.log(1+u*np.abs(voices))/np.log(1+u)
    return rs

def get_voices_labels(voices: List[List[int]], label_count=256):
    voices_with_pads, _ = fill_voices_data_with_pads(voices)
    x = np.array(voices_with_pads)/(65536/2)
    x = np.sign(x)*np.log(1+label_count*np.abs(x))/np.log(1+label_count)
    step = 2 / label_count
    x = x // step + label_count // 2
    return x.astype(int)


