import os
import time
import sys
import math
import argparse
import json
from multiprocessing import Process, Queue

import numpy as np
import torch
from tqdm import tqdm
from nmt_model import NMT, Hypothesis
from typing import List, Tuple, Dict, Set, Union
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import load_voices, load_voices_files, split_source_with_pad, read_corpus_from_LJSpeech, batch_iter, get_voice_files_and_corpus, batch_iter_to_queue, batch_iter_to_queue2, load_train_data, pad_source
from vocab import Vocab, VocabEntry
from dataloader import DataLoader

import pylab
import librosa


def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(
                len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def beam_search(model: NMT, test_data_src: List[List[float]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[float]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(
                src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training:
        model.train(was_training)

    return hypotheses


def train(model_config, data_config, output_path, device,
          epoch_size, max_epoch, batch_size, repeats, 
          decade_rate, clip_grad, log_every, valid_every, learning_rate=0.0005):
    print('use device: %s' % device, file=sys.stderr)
    vocab = Vocab.load(data_config["vacab_file"])
    model = NMT(vocab=vocab, **model_config)
    model = model.to(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    data_config.pop("vacab_file", None)
    data_loader = DataLoader(**data_config)
    batch_queue, loss_queue = data_loader.load_train_data(
        epoch_size, max_epoch, batch_size, repeats, decade_rate)
    dev_data = data_loader.load_dev_data()



    hist_valid_scores = []
    train_losses = []
    train_iter = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0

    if os.path.isfile(output_path+ '/speech-to-text.model'):
        print('loading saved model...')
        params = torch.load(output_path+ '/speech-to-text.model', map_location=lambda storage, loc: storage)
        model.load_state_dict(params['state_dict'])
        print('restoring parameters of the optimizers', file=sys.stderr)
        optimizer.load_state_dict(torch.load(output_path+ '/speech-to-text.optim'))
        dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
        valid_metric = -dev_ppl
        hist_valid_scores.append(valid_metric)
        print("saved model ppl: ", dev_ppl)

    model.train()

    train_time = begin_time = time.time()
    epoch, voices, tgt_sents = batch_queue.get(True)
    while voices is not None and tgt_sents is not None:
        train_iter += 1
        optimizer.zero_grad()
        # print("received voices:", len(voices))
        # print("tgt_sents[0]:", len(tgt_sents[0]), tgt_sents[0])
        # print("tgt_sents[1]:", len(tgt_sents[1]), tgt_sents[1])
        optimizer.zero_grad()
        batch_size = len(voices)
        sample_losses = -model(voices, tgt_sents)
        batch_loss = sample_losses.sum()
        loss = batch_loss / batch_size
        loss.backward()

        # clip gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_grad)
        optimizer.step()

        batch_losses_val = batch_loss.item()
        report_loss += batch_losses_val
        cum_loss += batch_losses_val

        tgt_words_num_to_predict = sum(
            len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
        report_tgt_words += tgt_words_num_to_predict
        cum_tgt_words += tgt_words_num_to_predict
        report_examples += batch_size
        cum_examples += batch_size
        loss_queue.put(report_loss / report_examples)
        train_losses.append({'epoch': epoch,
                             'iter': train_iter,
                             'loss': report_loss / report_tgt_words,
                             'ppl': math.exp(report_loss / report_tgt_words),
                             'cum': cum_examples,
                             'speed': report_tgt_words / (time.time() - train_time)})

        if train_iter % log_every == 0:
            print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f '
                  'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                     report_loss / report_examples,
                                                                                     math.exp(
                                                                                         report_loss / report_tgt_words),
                                                                                     cum_examples,
                                                                                     report_tgt_words /
                                                                                     (time.time(
                                                                                     ) - train_time),
                                                                                     time.time() - begin_time), file=sys.stderr)

            train_time = time.time()
            report_loss = report_tgt_words = report_examples = 0.
        # perform validation
        if train_iter % valid_every == 0:
            print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                    cum_loss / cum_examples,
                                                                                    np.exp(cum_loss / cum_tgt_words),
                                                                                    cum_examples), file=sys.stderr)

            cum_loss = cum_examples = cum_tgt_words = 0.
            valid_num += 1

            print('begin validation ...', file=sys.stderr)

            # compute dev. ppl and bleu
            dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
            valid_metric = -dev_ppl

            print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

            is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
            hist_valid_scores.append(valid_metric)

            if is_better:
                patience = 0
                print('save currently the best model to [%s]' % output_path, file=sys.stderr)
                model.save(output_path+ '/speech-to-text.model')
                torch.save(optimizer.state_dict(), output_path + '/speech-to-text.optim')

        epoch, voices, tgt_sents = batch_queue.get(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')

    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    # print("start training with config:", )
    # print(json.dumps(config, indent=4, sort_keys=True))

    train(config["model_config"], config["data_config"],
          **config["train_config"])

    # device = torch.device("cuda:0")
    # device = torch.device("cpu")
    # print('use device: %s' % device, file=sys.stderr)
