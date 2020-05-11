#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import os
import math
import os.path
import time
from typing import List
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa  # for audio processing
from multiprocessing import Process, Queue
import re
from typing import Callable, Any, Iterable


def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()`
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = 21

    # YOUR CODE HERE for part 1f
    # TODO:
    # Perform necessary padding to the sentences in the batch similar to the pad_sents()
    # method below using the padding character from the arguments. You should ensure all
    # sentences have the same number of words and each word has the same number of
    # characters.
    # Set padding words to a `max_word_length` sized vector of padding characters.
    ###
    # You should NOT use the method `pad_sents()` below because of the way it handles
    # padding and unknown words.
    sents_padded = []
    max_sentence_length = 0
    for sentence in sents:
        sentence_with_padded_words = []
        sentence_length = len(sentence)
        if sentence_length > max_sentence_length:
            max_sentence_length = sentence_length
        for word in sentence:
            word_length = len(word)
            if word_length >= max_word_length:
                sentence_with_padded_words.append(word[:max_word_length])
            else:
                sentence_with_padded_words.append(
                    word+[char_pad_token]*(max_word_length-word_length))
        sents_padded.append(sentence_with_padded_words)

    for i in range(len(sents_padded)):
        sents_padded[i] = sents_padded[i] + [[char_pad_token] *
                                             max_word_length]*(max_sentence_length-len(sents_padded[i]))

    # END YOUR CODE

    return sents_padded


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    # COPY OVER YOUR CODE FROM ASSIGNMENT 4
    maxlen = 0
    for sentence in sents:
        length = len(sentence)
        if length > maxlen:
            maxlen = length
    for sentence in sents:
        padded_sentence = sentence + [pad_token] * (maxlen - len(sentence))
        sents_padded.append(padded_sentence)

    # END YOUR CODE FROM ASSIGNMENT 4

    return sents_padded


def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def read_corpus_from_LJSpeech(file_path, source, line_num=-1):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
        each line is begin with 11 charactors wav file name:  'LJ001-0004|'
        the rename text is the speech of the wav.                  
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    line_count = 0
    for line in open(file_path):
        sent_info = line.split('|')
        voice_name = sent_info[0]
        sent = re.sub('[,";:\?\(\)]', '', sent_info[-1])\
            .lower()\
            .replace("-- ", "")\
            .replace("-", " ")\
            .replace("'s ", " 's ")\
            .replace(". ", " ")\
            .strip()\
            .split(' ')
        last_char = sent[-1][-1]
        if last_char in ['.', ';', ","]:
            sent[-1] = sent[-1][:-1]
        #     sent = sent + [last_char]

        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        yield (voice_name, sent)
        line_count += 1
        if line_count == line_num:
            break


def get_voice_files_and_corpus(voice_path: str, voice_num=-1) -> Tuple[List[str], List[List[str]]]:
    corpus_map = read_corpus_from_LJSpeech(
        voice_path + '/metadata.csv', 'tgt', voice_num)
    voice_files = []
    corpus = []
    for voice_file, sent in corpus_map:
        voice_files.append(voice_path+'/'+voice_file+'.wav')
        corpus.append(sent)
    return voice_files, corpus


def load_train_data(voice_path: str, data_size: int, epoch_size: int, data_queue: Queue, repeat=1, decade_rate=0.5):
    print("loading train data ...")
    sample_rate = 22000
    resample_rate = 8000
    data = read_corpus_from_LJSpeech(
        voice_path + '/metadata.csv', 'tgt', data_size)
    voices = []
    corpus = []
    data_count = 0
    epoch_count = 0

    remaining_records = int((1 - decade_rate) * epoch_size // 1)
    print("remaining train data length:", remaining_records)
    corpus_map = []

    for voice_file, sent in data:
        corpus_map.append((voice_file, sent))
    corpus_index_array = list(range(len(corpus_map)))
    for rd in range(repeat):
        print("pushing new round train data:", rd)

        np.random.shuffle(corpus_index_array)
        for idx in corpus_index_array:
            voice_file, sent = corpus_map[idx]
            while not data_queue.empty():
                time.sleep(3)
            voice_file = voice_path+'/'+voice_file+'.wav'
            if not os.path.isfile(voice_file):
                continue
            samples, sample_rate = librosa.load(voice_file, sr=sample_rate)
            voices.append(librosa.resample(
                samples, sample_rate, resample_rate))
            corpus.append(sent)
            epoch_count = epoch_count + 1
            data_count = data_count + 1
            if epoch_count == epoch_size:
                print("push new train data ...")
                train_data = list(zip(voices, corpus))
                data_queue.put(train_data, True)
                index_array = list(range(epoch_size))
                voices = []
                corpus = []
                epoch_count = remaining_records
                for idx in index_array[:remaining_records]:
                    voices.append(train_data[idx][0])
                    corpus.append(train_data[idx][1])


    print("all train data has been loaded")
    data_queue.put(None, True)
    return


def get_voice_files_and_corpus_by_indexes(voice_path: str, indexes) -> Tuple[List[str], List[List[str]]]:
    corpus_map = read_corpus_from_LJSpeech(voice_path + '/metadata.csv', 'tgt')
    voice_files = []
    corpus = []
    index_count = len(indexes)
    index_pos = 0
    corpus_pos = 0
    for voice_file, sent in corpus_map:
        if indexes[index_pos] == corpus_pos:
            index_pos += 1
            if not os.path.isfile(voice_file):
                continue
            yield (voice_path+'/'+voice_file+'.wav', sent)
            if index_pos >= index_count:
                break
        corpus_pos += 1


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


def batch_iter_to_queue(data, batch_queue, epoch_num, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    for epoch in range(epoch_num):
        # print("epoch:", epoch, "started")
        batch_num = math.ceil(len(data) / batch_size)
        index_array = list(range(len(data)))

        if shuffle:
            np.random.shuffle(index_array)

        for i in range(batch_num):
            indices = index_array[i * batch_size: (i + 1) * batch_size]
            examples = [data[idx] for idx in indices]

            examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
            voice_files = [e[0] for e in examples]
            # voices = load_voices_files(voice_files, sample_rate, resample_rate)
            voices = voice_files
            tgt_sents = [e[1] for e in examples]
            batch_queue.put((epoch, voices, tgt_sents))
    batch_queue.put((None, None, None))


def batch_iter_to_queue2(data_queue, batch_queue, loss_queue, epoch_num, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    print("geting train data ...")
    data = data_queue.get(True)

    train_index = 0
    while data is not None:
        train_index += 1
        print("start new training %d: data(size = %s) in %d epoches : ..." %
              (train_index, len(data), epoch_num))
        for epoch in range(epoch_num):
            # print("epoch:", epoch, "started")
            batch_num = math.ceil(len(data) / batch_size)
            index_array = list(range(len(data)))

            if shuffle:
                np.random.shuffle(index_array)
            loss_sum = 0.0

            for i in range(batch_num):
                indices = index_array[i * batch_size: (i + 1) * batch_size]
                examples = [data[idx] for idx in indices]

                examples = sorted(
                    examples, key=lambda e: len(e[0]), reverse=True)
                voice_files = [e[0] for e in examples]
                # voices = load_voices_files(voice_files, sample_rate, resample_rate)
                voices = voice_files
                tgt_sents = [e[1] for e in examples]
                # print("push batch data ...")
                batch_queue.put((epoch, voices, tgt_sents), True)
                loss_sum = loss_sum + loss_queue.get()
            if loss_sum/batch_num < 0.5:
                break

        print("geting train data ...")
        data = data_queue.get(True)
        if data is not None:
            print("recieved train data (size = %d) ..." % len(data))
        else:
            print("recieved no train data")

    batch_queue.put((None, None, None))


def read_voice(voice_file, sample_rate, resample_rate=8000, chunk_size=2048, pad_value=0.0):
    samples, sample_rate = librosa.load(voice_file, sr=sample_rate)
    samples = librosa.resample(samples, sample_rate, resample_rate)
    sample_len = len(samples)
    chunk_num, left = divmod(sample_len, chunk_size)
    speech_chunks = []
    for i in range(chunk_num):
        speech_chunks.append(samples[i*chunk_size:(i+1)*chunk_size])
    if left > 0:
        speech_chunks.append(
            np.concatenate((samples[chunk_num*chunk_size:], [pad_value] * (chunk_size-left))))
    return speech_chunks


def load_voices(voice_path, sample_rate, resample_rate=8000, voice_num=-1):
    voices = []
    for file in os.listdir(voice_path):
        if file.endswith(".wav"):
            voice_file = os.path.join(voice_path, file)
            samples, sample_rate = librosa.load(voice_file, sr=sample_rate)
            samples = librosa.resample(samples, sample_rate, resample_rate)
            voices.append(samples)
            if len(voices) == voice_num:
                break
    return voices


def load_voices_files(voice_files, sample_rate, resample_rate=8000, voice_num=-1):
    voices = []
    for voice_file in voice_files:
        samples, sample_rate = librosa.load(voice_file, sr=sample_rate)
        samples = librosa.resample(samples, sample_rate, resample_rate)
        voices.append(samples)
        if len(voices) == voice_num:
            break
    return voices


def split_source_with_pad(source: List[List[float]], chunk_size=2048, max_chunk=40, pad_value=0.0) -> (np.ndarray, List[int]):
    """ split source with pad.

        @param source (List[List[float]]): source to be splitted
        @param chunk_size (int): split Size
        @param max_chunk (int): max number of chunk
        @param pad_value (float): the pad value
        """
    splited_source = []
    lengths = []
    for data in source:
        sample_len = len(data)
        chunk_num, left = divmod(sample_len, chunk_size)
        if left == 0:
            lengths.append(chunk_num)
        else:
            lengths.append(chunk_num+1)
        padded_data = np.concatenate(
            (data, [pad_value] * (chunk_size*max_chunk - sample_len)))
        chunks = np.split(padded_data, max_chunk)
        splited_source.append(chunks)
    return np.array(splited_source), lengths


def split_voices_with_pad(voices: List[List[float]], chunk_size=1024, max_chunk=80, pad_value=0.0) -> (np.ndarray, List[int]):
    """ split source with pad.

        @param source (List[List[float]]): source to be splitted
        @param chunk_size (int): split Size
        @param max_chunk (int): max number of chunk
        @param pad_value (float): the pad value
        """
    splited_source = []
    lengths = []
    for voice in voices:
        splited_voice = split_voice_with_pad(voice, chunk_size, pad_value)
        chunk_num = len(splited_voice)
        if chunk_num < max_chunk:
            lengths.append(chunk_num)
            padded_data = splited_voice + \
                [[pad_value]*chunk_size]*(max_chunk - chunk_num)
        else:
            lengths.append(max_chunk)
            padded_data = splited_voice[:max_chunk]
        splited_source.append(padded_data)
    return np.array(splited_source), lengths


def split_voice_with_pad(voice: List[float], chunk_size=1024, pad_value=0.0) -> List[List[float]]:
    """ split voice with pad.

        @param voice (List[float]): samples to be splitted
        @param chunk_size (int): split Size
        @param max_chunk (int): max number of chunk
        @param pad_value (float): the pad value
        """
    chunks = []
    peek_length = 10

    def isGap(current_chunk, idx):
        sum = 0
        length = 0
        for i in range(peek_length):
            if idx+i < len(voice):
                length = length + 1
                sum = sum + abs(voice[idx+i])
        next_avg = sum/length
        current_chunk['avg'] = current_chunk['sum'] / current_chunk['len']
        return next_avg < current_chunk['avg'] * 0.5 or next_avg > current_chunk['avg'] * 2 or current_chunk['len'] >= chunk_size

    def withPads(chunk):
        chunk_len = chunk['len']
        if chunk_len == chunk_size:
            return chunk['data']
        return chunk['data'].tolist() + [pad_value] * (chunk_size - chunk_len)

    current_chunk = {
        'start': 0,
        'end': 0,
        'sum': 0.0,
        'len': 0,
    }
    for i in range(len(voice)):
        if current_chunk['len'] > 1000 and isGap(current_chunk, i):
            current_chunk['end'] = i
            current_chunk['data'] = voice[current_chunk['start']:current_chunk['end']]
            chunks.append(withPads(current_chunk))
            current_chunk = {
                'start': i,
                # 'data': [voice[i]],
                'sum': voice[i],
                'len': 1,
            }
        else:
            # current_chunk['data'].append(voice[i])
            current_chunk['sum'] = current_chunk['sum'] + abs(voice[i])
            current_chunk['len'] = current_chunk['len'] + 1
            # current_chunk['avg'] = current_chunk['sum'] / current_chunk['len']
    if current_chunk['len'] > 500:
        current_chunk['data'] = voice[current_chunk['start']:]
        chunks.append(withPads(current_chunk))

    return chunks


def load_voices_with_pad(voice_path, sample_rate, resample_rate=8000, chunk_size=2048, chunk_num=40, pad_value=0.0):
    voices = []
    lengths = []
    for file in os.listdir(voice_path):
        if file.endswith(".wav"):
            voice_data = read_voice(os.path.join(
                voice_path, file), sample_rate, resample_rate, chunk_size)
            if len(voice_data) >= chunk_num:
                voices.append(voice_data[:chunk_num])
                lengths.append(chunk_num)
            else:
                lengths.append(len(voice_data))
                for _ in range(chunk_num-len(voice_data)):
                    voice_data.append([pad_value] * chunk_size)
                voices.append(voice_data)

            voices.append(voice_data)
            if len(voices) == 100:
                break
    return np.array(voices), lengths
