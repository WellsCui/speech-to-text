from multiprocessing import Process, Queue
import librosa
from utils import load_voices, load_voices_files, split_source_with_pad, read_corpus_from_LJSpeech, batch_iter, get_voice_files_and_corpus, batch_iter_to_queue, batch_iter_to_queue2, load_train_data
import os
import numpy as np

class DataLoader(object):
    def __init__(self, train_file, dev_file, waves_path, sampling_rate, n_mels, mel_fmax):
        self.train_file = train_file
        self.dev_file = dev_file
        self.waves_path = waves_path
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.mel_fmax = mel_fmax
        self.data_queue = Queue(2)
        self.batch_queue = Queue(2)
        self.loss_queue = Queue(2)
        self.compressing_queue = Queue(64)

    def load_train_data(self, epoch_size, max_epoch, batch_size, repeats, decade_rate) -> (Queue, Queue):
        self.train_data_to_queue_process = Process(target=load_train_data, args=(
            self.train_file, self.waves_path, -1, epoch_size, self.data_queue, self.loadVoice_new, repeats, decade_rate))
        self.train_data_to_queue_process.start()
        self.batch_iter_to_queue_process = Process(target=batch_iter_to_queue2, args=(
            self.data_queue, self.batch_queue, self.loss_queue, max_epoch, batch_size, True))
        self.batch_iter_to_queue_process.start()
        self.save_proccessed_data_process = Process(target=self.save_proccessed_data)
        self.save_proccessed_data_process.start()
        return self.batch_queue, self.loss_queue

    def loadVoice(self, voice_file):
        wav, sr = librosa.load(voice_file)
        S = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_mels=self.n_mels, fmax=self.mel_fmax)
        return S

    def save_proccessed_data(self):
        print("start saving proccessed data ...")
        voice_file, data = self.compressing_queue.get(True)
        while voice_file is not None and data is not None:
            np.savez_compressed(self.waves_path+"/"+voice_file, melspectrogram=data)
            # print("saved compressed voice file:", voice_file)
            voice_file, data = self.compressing_queue.get(True)
        print("end saving proccessed data ...")


    def loadVoice_new(self, voice_file):
        melspectrogram_file = self.waves_path+'/'+voice_file+'.npz'
        if os.path.isfile(melspectrogram_file):
            print("loading compressed voice file:", voice_file)
            return np.load(melspectrogram_file)["melspectrogram"]
        wav, sr = librosa.load(self.waves_path+'/'+voice_file+'.wav')
        S = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_mels=self.n_mels, fmax=self.mel_fmax)
        self.compressing_queue.put((voice_file, S), True)
        return S


    def load_dev_data(self):
        dev_files, dev_corpus = get_voice_files_and_corpus(
            self.dev_file, self.waves_path)
        voices = []
        for voice_file in dev_files:
            voices.append(self.loadVoice(voice_file))
        return list(zip(voices, dev_corpus))
