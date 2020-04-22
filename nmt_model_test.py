"""
CS224N 2018-19: Homework 5
run.py: Run Script for Simple NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>

Usage:
    run.py train [options]
    run.py decode MODEL_PATH OUTPUT_FILE [options]

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 300]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 100]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --no-char-decoder                       do not use the character decoder
"""

from docopt import docopt
import numpy as np
import torch
import time
import sys
import math
from tqdm import tqdm
from nmt_model import NMT, Hypothesis
from typing import List, Tuple, Dict, Set, Union
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from multiprocessing import Process, Queue
from utils import load_voices, load_voices_files, split_source_with_pad, read_corpus_from_LJSpeech, batch_iter, get_voice_files_and_corpus, batch_iter_to_queue, batch_iter_to_queue2, load_train_data

from vocab import Vocab, VocabEntry




def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    sample_rate = 22000
    resample_rate = 8000
    train_records = 8
    max_epoch = 10
    vocab = Vocab.load('dataset/vocab_full.json')
    # train_voices_files, corpus = get_voice_files_and_corpus('dataset/train/wavs', train_records)
    # voices = load_voices_files(train_voices_files, sample_rate, resample_rate)
    # train_data = list(zip(voices, corpus))

    dev_files, dev_corpus = get_voice_files_and_corpus('dataset/dev', 2)
    dev_data = list(zip(load_voices_files(dev_files, sample_rate, resample_rate), dev_corpus))

    epoch_size = 4
    train_batch_size = 2

    clip_grad = 5.0
    valid_niter = 100
    log_every = 10
    model_save_path = 'model.bin'

    model = NMT(embed_size=1024, hidden_size=2048, vocab=vocab)
    model.train()
    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')
    data_queue = Queue()
    batch_queue = Queue(1)
    loss_queue = Queue(2)

    train_data_to_queue_process = Process(target=load_train_data, args=('dataset/train', train_records, epoch_size, data_queue))
    train_data_to_queue_process.start()

    batch_iter_to_queue_process = Process(target=batch_iter_to_queue2, args=(data_queue, batch_queue, loss_queue, max_epoch, train_batch_size, True))
    batch_iter_to_queue_process.start()
    epoch, voices, tgt_sents = batch_queue.get(True)
    current_epoch = -1

    while voices is not None and tgt_sents is not None:

        train_iter += 1
        optimizer.zero_grad()
        
        # voices = load_voices_files(voice_files, sample_rate, resample_rate)
        # voices = voice_files
        batch_size = len(voices)

        example_losses = -model(voices, tgt_sents) # (batch_size,)
        batch_loss = example_losses.sum()
        loss = batch_loss / batch_size

        loss.backward()

        # clip gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        batch_losses_val = batch_loss.item()
        report_loss += batch_losses_val
        cum_loss += batch_losses_val

        tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
        report_tgt_words += tgt_words_num_to_predict
        cum_tgt_words += tgt_words_num_to_predict
        report_examples += batch_size
        cum_examples += batch_size
        loss_queue.put(report_loss / report_examples)

        if train_iter % log_every == 0:
            print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                    report_loss / report_examples,
                                                                                    math.exp(report_loss / report_tgt_words),
                                                                                    cum_examples,
                                                                                    report_tgt_words / (time.time() - train_time),
                                                                                    time.time() - begin_time), file=sys.stderr)

            train_time = time.time()
            report_loss = report_tgt_words = report_examples = 0.

        # perform validation
        if train_iter % valid_niter == 0:
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
                print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                model.save(model_save_path)

                # also save the optimizers' state
                torch.save(optimizer.state_dict(), model_save_path + '.optim')
            elif patience < 10:
                patience += 1
                print('hit patience %d' % patience, file=sys.stderr)

                if patience == 5:
                    num_trial += 1
                    print('hit #%d trial' % num_trial, file=sys.stderr)
                    if num_trial == 3:
                        print('early stop!', file=sys.stderr)
                        exit(0)

                    # decay lr, and restore from previously best checkpoint
                    lr = optimizer.param_groups[0]['lr'] * 0.5
                    print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                    # load model
                    params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                    model.load_state_dict(params['state_dict'])
                    model = model.to(device)

                    print('restore parameters of the optimizers', file=sys.stderr)
                    optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                    # set new lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    # reset patience
                    patience = 0
            
        epoch, voices, tgt_sents = batch_queue.get()
    batch_iter_to_queue_process.join()
    train_data_to_queue_process.join()
    

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


def decode(args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    # print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)

    voices_files, test_data_tgt = get_voice_files_and_corpus('dataset/dev', 2)
    test_data_src = load_voices(voices_files, 22000, 8000)


    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'],
                     no_char_decoder=args['--no-char-decoder'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    top_hypotheses = [hyps[0] for hyps in hypotheses]
    bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
    print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


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


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check pytorch version
    # assert(torch.__version__ == "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
