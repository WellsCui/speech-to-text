from multiprocessing import Process, Queue
import librosa
from utils import load_voices, load_voices_files, split_source_with_pad, read_corpus_from_LJSpeech, batch_iter, get_voice_files_and_corpus, batch_iter_to_queue, batch_iter_to_queue2, load_train_data


class DataLoader(object):
    def __init__(self, train_file, dev_file, waves_path, sampling_rate, n_mels, mel_fmax):
        self.train_file = train_file
        self.dev_file = dev_file
        self.waves_path = waves_path
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.mel_fmax = mel_fmax
        self.data_queue = Queue(1)
        self.batch_queue = Queue(1)
        self.loss_queue = Queue(2)

    def load_train_data(self, epoch_size, max_epoch, batch_size, repeats, decade_rate) -> (Queue, Queue):
        self.train_data_to_queue_process = Process(target=load_train_data, args=(
            self.train_file, self.waves_path, -1, epoch_size, self.data_queue, self.loadVoice, repeats, decade_rate))
        self.train_data_to_queue_process.start()
        self.batch_iter_to_queue_process = Process(target=batch_iter_to_queue2, args=(
            self.data_queue, self.batch_queue, self.loss_queue, max_epoch, batch_size, True))
        self.batch_iter_to_queue_process.start()
        return self.batch_queue, self.loss_queue

    def loadVoice(self, voice_file):
        wav, sr = librosa.load(voice_file)
        S = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_mels=self.n_mels, fmax=self.mel_fmax)
        return S

    def load_dev_data(self):
        dev_files, dev_corpus = get_voice_files_and_corpus(
            self.dev_file, self.waves_path)
        voices = []
        for voice_file in dev_files:
            voices.append(self.loadVoice(voice_file))
        return list(zip(voices, dev_corpus))
