import numpy as np
import math
import sys
import time
import datetime
import os
import copy

from transformers import BertModel, BertConfig, AdamW

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import pytorch_lightning

import prepare_data
import pickle
import argparse


np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description='')

# training setup
parser.add_argument('--dict-file', type=str, default='/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/predict-middle-notes/dictionary_paddedLM.pickle')
parser.add_argument('--data-file', type=str, default='/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/predict-middle-notes/worded_data.pickle')
parser.add_argument('--data-file-2track', type=str, default='/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/predict-middle-notes-keep-melody/worded_data_melody_at_beginning.pickle')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--save-path', type=str, default="/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/paddedLM/trained-models/partial-target-test")
parser.add_argument('--batch-size', type=int, default=6)
parser.add_argument('--target-max-percent', type=float, default=0.2, help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
parser.add_argument('--n-step-bars', type=int, default=8, help='how many bars to step before next training data fetching (the smaller the more training data)')
parser.add_argument('--max-seq-len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
parser.add_argument('--train-epochs', type=int, default=2000, help='number of training epochs')
parser.add_argument('--init-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--mask-len', type=int, default=110, help='Default length for blank')
parser.add_argument('--seed', type=int, default=np.random.randint(100000), help='Random seed')

# for prediction phase
parser.add_argument('--test-data-file', type=str, default='/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/predict-middle-notes/worded_data.pickle')
parser.add_argument('--ckpt-path', type=str, default="/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/predict-middle-notes/trained-models/short-target-mapping/loss34.ckpt")
parser.add_argument('--song-idx', type=int, default=170)

args = parser.parse_args()

pytorch_lightning.utilities.seed.seed_everything(seed=args.seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# configuration = BertConfig().from_dict({
#   "_name_or_path": "paddedLM-predict-middle-notes",
#   # "bos_token_id": 10000,
#   "clamp_len": -1,
#   # "d_head": 64,
#   "d_inner": 3072,
#   "d_model": 768,
#   "dropout": 0.1,
#   # "end_n_top": 5,
#   # "eos_token_id": 2,
#   "ff_activation": "gelu",
#   "initializer_range": 0.02,
#   "layer_norm_eps": 1e-12,
#   # "mem_len": 0, # null
#   "n_head": 8,  # 12 originally
#   "n_layer": 12,
#   # "pad_token_id": 10000,
#   # "reuse_len": None, # null,
#   # "same_length": False,
#   "start_n_top": 5,
#   "summary_activation": "tanh",
#   "summary_last_dropout": 0.1,
#   # "summary_type": "last",
#   # "summary_use_proj": True,
#   # "untie_r": True,
#   # "use_mems_eval": True,
#   # "use_mems_train": True,
#   # "vocab_size": 32000
# })
# configuration = BertConfig(max_position_embeddings=args.max_seq_len + args.mask_len,
#                            position_embedding_type="relative_key_query")
configuration = BertConfig(max_position_embeddings=1024,
                           position_embedding_type="relative_key_query")

def show_events(data, word2event, output=None, target=None, loss_mask=None):
    print("\n\n" + "=" * 80)
    tes = []    # tuple events
    # for seq_idx, (e, output, target) in enumerate(zip(data, output, target)):
    for seq_idx, (e, target) in enumerate(zip(data, target)):
        e = [word2event[etype][e[i]] for i, etype in enumerate(word2event)]
        # output = [word2event[etype][output[i]] for i, etype in enumerate(word2event)]
        target = [word2event[etype][target[i]] for i, etype in enumerate(word2event)]
        print("     ", e)
        # print("[OUT]", output)
        print("[TGT]", target, end='  ')

        if loss_mask != None and loss_mask[seq_idx] == 1:
            print('[target]\n')
        else:
            print('\n')

def to_midi_absolute_bar(data, word2event, path_outfile):
    tes = []    # tuple events
    for e in data:
        e = [word2event[etype][e[i]] for i, etype in enumerate(word2event)]

        skip = False
        for i in range(len(e2w)):
            if 'MASK' in e[i] or 'EOS' in e[i]:
                skip = True
        if skip:
            continue

        te = prepare_data.GroupEvent(Tempo=int(e[0].split(' ')[1]),
                                     Bar=int(e[1].split(' ')[1]),
                                     Position=e[2].split(' ')[1],
                                     Pitch=int(e[3].split(' ')[1]),
                                     Duration=int(e[4].split(' ')[1]),
                                     Velocity=int(e[5].split(' ')[1])
                                     )
        tes.append(te)

    prepare_data.tuple_events_to_midi(tes, path_outfile)
# --- write tool --- #
def to_midi(data, word2event, path_outfile):
    tes = []    # tuple events
    cur_bar = 0
    for i, e in enumerate(data):
        e_word = copy.deepcopy(e)
        e = [word2event[etype][e[i]] for i, etype in enumerate(word2event)]

        skip = False
        for i, etype in enumerate(word2event):
            if 'PAD' in e[i] or 'BOS' in e[i] or 'BLK' in e[i] or 'MASK' in e[i] or 'EOS' in e[i]:
                skip = True
        if skip:
            continue

        if e_word[1] == 1:  # new bar event
            cur_bar += 1

        te = prepare_data.GroupEvent(Tempo=int(e[0].split(' ')[1]),
                                     Bar=cur_bar,
                                     Position=e[2].split(' ')[1],
                                     Pitch=int(e[3].split(' ')[1]),
                                     Duration=int(e[4].split(' ')[1]),
                                     Velocity=int(e[5].split(' ')[1])
                                     )
        tes.append(te)

    prepare_data.tuple_events_to_midi(tes, path_outfile)

########################################
# search strategy: temperature (re-shape)
########################################
def temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs

########################################
# search strategy: nucleus (truncate)
########################################
def nucleus(probs, p):
    # print('probs:', probs)
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    # print('probs:', probs)
    # print(after_threshold)
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3] # just assign a value
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# BERT model: similar approach to "felix"
class BertForPredictingMiddleNotes(torch.nn.Module):
    def __init__(self, bertConfig, e2w, w2e):
        super(BertForPredictingMiddleNotes, self).__init__()
        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.bertConfig = bertConfig
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        # token types: [Tempo, Bar, Position, Pitch, Duration, Velocity]
        self.n_tokens = []
        for key in e2w:
            self.n_tokens.append(len(e2w[key]))
        # self.n_tokens[1] = 4
        self.emb_sizes = [256, 256, 256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.tempo_pad_word = self.e2w['Tempo']['Tempo <PAD>']

        self.eos_word = torch.Tensor([self.e2w[etype]['%s <EOS>' % etype] for etype in self.e2w]).long().to(device)
        self.bos_word = torch.Tensor([self.e2w[etype]['%s <BOS>' % etype] for etype in self.e2w]).long().to(device)

        self.blk_word_np = np.array([self.e2w[etype]['%s <BLK>' % etype] for etype in self.e2w], dtype=np.long)
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.e2w], dtype=np.long)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.e2w], dtype=np.long)

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.e2w):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types to feed into transformer-XL
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)

        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(self.e2w):
            self.proj.append(nn.Linear(bertConfig.d_model, self.n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)


    def forward(self, input_ids, attn_mask=None):
        # convert input_ids into embeddings and merge them through linear layer
        embs =[]
        for i, key in enumerate(self.e2w):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        # feed to transformer-XL
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask)
        y = y.last_hidden_state

        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))

        return ys


    def compute_loss(self, predict, target, loss_mask, meaningful_loss_mask=None):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        if meaningful_loss_mask is not None:
            meaningful_loss = (meaningful_loss_mask * loss).detach().cpu().numpy()
            meaningful_loss = np.sum(meaningful_loss) / torch.sum(meaningful_loss_mask).cpu().item()
        loss = torch.sum(loss) / torch.sum(loss_mask)

        if meaningful_loss_mask is None:
            return loss
        else:
            return loss, meaningful_loss

    def train(self, training_data=None, n_epochs=None):
        os.makedirs(args.save_path, exist_ok=True)
        path_saved_ckpt = os.path.join(args.save_path, 'loss')

        start_end = np.zeros((len(training_data), 2))
        for i in range(len(training_data)):
            # calculate the index of start of bar6 and the end of bar9
            training_data[i] = np.array(training_data[i])
            start_end[i][0] = np.nonzero(training_data[i][:, 1] == 6)[0][0]
            start_end[i][1] = np.nonzero(training_data[i][:, 1] == 9)[0][-1]

            # absolute bar -> relative bar
            training_data[i][1:, 1] = training_data[i][1:, 1] - training_data[i][:-1, 1]
            training_data[i][:, 1][training_data[i][:, 1] > 1] = 1  # avoid bug when there are empty bars


        start_time = time.time()
        optimizer = AdamW(self.parameters(), lr=args.init_lr, weight_decay=0.01)
        num_batches = len(training_data) // (args.batch_size)
        for epoch in range(args.train_epochs):
            total_losses = 0
            total_meaningful_losses = 0
            for train_iter in range(num_batches):
                # input_ids = torch.from_numpy(training_data[train_iter*args.batch_size:(train_iter+1)*args.batch_size]).to(device)
                ori_seq_batch = training_data[train_iter*args.batch_size:(train_iter+1)*args.batch_size]
                start_end_batch = start_end[train_iter * args.batch_size : (train_iter + 1) * args.batch_size]

                # decide the range to be predicted: `target_start` to `target_start + target_len`
                # target_starts = [np.random.randint(0, int(len(seq) * (1 - args.target_max_percent))) for seq in ori_seq_batch]
                # target_starts = [np.random.randint(0, int(len(seq) - 20)) for seq in ori_seq_batch]
                # target_lens = [max(2, np.random.randint(int(len(seq) * args.target_max_percent / 2), int(len(seq) * args.target_max_percent)))
                #                for seq in ori_seq_batch]
                # target_lens = [20 for seq in ori_seq_batch]
                # target_lens = [np.random.randint(int((end - start) * 0.5), end - start + 1) for (start, end) in start_end_batch]
                target_lens = [int(end - start) for (start, end) in start_end_batch]
                # target_lens = [8 for (start, end) in start_end_batch]
                # target_starts = [np.random.randint(start, end - target_len + 1) for (start, end), target_len in zip(start_end_batch, target_lens)]
                target_starts = [int(start) for (start, end), target_len in zip(start_end_batch, target_lens)]
                target_ends = [s + l for s, l in zip(target_starts, target_lens)]

                # prepare input for training
                # [pre-mask-x  mask post-mask-x]
                input_ids = np.full((args.batch_size, args.max_seq_len+args.mask_len, len(self.e2w)), self.pad_word_np)
                for b in range(args.batch_size):
                    if target_starts[b] != 0:
                        input_ids[b, 0:target_starts[b]] = ori_seq_batch[b][0:target_starts[b]] # pre-mask x
                    input_ids[b, target_starts[b]:target_starts[b]+args.mask_len] = self.mask_word_np   # mask
                    if target_ends[b] != len(ori_seq_batch[b]):
                        input_ids[b, target_starts[b]+args.mask_len:target_starts[b]+args.mask_len+len(ori_seq_batch[b])-target_ends[b]] = \
                                ori_seq_batch[b][target_ends[b]:]    # post-mask x
                input_ids = torch.from_numpy(input_ids).to(device)

                # pseudo data
                # target_starts = [np.random.randint(0, int(len(seq) - 20)) for seq in ori_seq_batch]
                # target_lens = [20 for seq in ori_seq_batch]
                # target_ends = [s + l for s, l in zip(target_starts, target_lens)]
                # input_ids = np.arange(args.max_seq_len + args.mask_len)
                # input_ids = np.tile(input_ids, (args.batch_size, len(self.e2w), 1))
                # input_ids = np.transpose(input_ids, (0, 2, 1))
                # target = np.copy(input_ids[:])
                # for b in range(args.batch_size):
                #     input_ids[b, target_starts[b]:target_starts[b]+args.mask_len] = self.mask_word_np   # mask
                # input_ids = torch.from_numpy(input_ids).to(device)

                # avoid attend to pad word
                attn_mask = (input_ids[:, :, 0] != self.tempo_pad_word).float()

                y = self.forward(input_ids, attn_mask)

                # get the most likely choice with max
                # outputs = []
                # for i, etype in enumerate(self.e2w):
                #     output = np.argmax(y[i].cpu().detach().numpy(), axis=-1)
                #     outputs.append(output)
                # outputs = np.stack(outputs, axis=-1)

                # reshape (b, s, f) -> (b, f, s)
                for i, etype in enumerate(self.e2w):
                    y[i] = y[i][:, ...].permute(0, 2, 1)


                # calculate losses
                target = np.zeros(shape=input_ids.shape, dtype=np.long)
                loss_mask = torch.zeros(args.batch_size, args.max_seq_len + args.mask_len)
                meaningful_loss_mask = torch.zeros(args.batch_size, args.max_seq_len + args.mask_len).to(y[0].device)
                for b in range(args.batch_size):
                    target[b, target_starts[b]:target_ends[b]] = ori_seq_batch[b][target_starts[b]:target_ends[b]]
                    target[b, target_ends[b]:] = self.blk_word_np
                    loss_mask[b, target_starts[b]:target_starts[b]+args.mask_len] = 1
                    meaningful_loss_mask[b, target_starts[b] : target_ends[b]] = 1
                target = torch.from_numpy(target)
                losses = []
                meaningful_losses = []
                for i, etype in enumerate(self.e2w):
                    # losses.append(self.compute_loss(y[i], target[..., i].to(device), loss_mask.to(device)))
                    loss, meaningful_loss = self.compute_loss(y[i], target[..., i].to(device), loss_mask.to(device), meaningful_loss_mask=meaningful_loss_mask)
                    losses.append(loss)
                    meaningful_losses.append(meaningful_loss)
                total_loss = sum(losses) / len(self.e2w)
                total_meaningful_loss = sum(meaningful_losses) / len(self.e2w)

                # show_events(input_ids[0].cpu().detach().numpy(), self.w2e, target=target[0].cpu().detach().numpy(), loss_mask=loss_mask[0])
                # show_events(input_ids[1].cpu().detach().numpy(), self.w2e, output=outputs[1], target=target[1].cpu().detach().numpy(), loss_mask=loss_mask[1])

                # udpate
                self.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.parameters(), 3.0)
                optimizer.step()


                # acc
                sys.stdout.write('{}/{} | Loss: {:06f} | {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
                    train_iter, num_batches, total_meaningful_loss, *meaningful_losses))
                losses = list(map(float, losses))
                total_losses += total_loss.item()
                total_meaningful_losses += total_meaningful_loss

            runtime = time.time() - start_time
            print('epoch: {}/{} | Loss: {} | time: {}'.format(
                epoch, n_epochs, total_meaningful_losses/num_batches, str(datetime.timedelta(seconds=runtime))))
            print('    > loss: {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}'.format(*meaningful_losses))


            loss =  total_meaningful_losses / num_batches
            if 0.4 < loss <= 0.8:
                fn = int(loss * 10)
                fn = fn * 10
                torch.save(self.state_dict(), path_saved_ckpt + str(fn) + '.ckpt')
            elif 0.1 < loss <= 0.4:
                fn = int(loss * 100)
                if fn % 2 == 0:
                    torch.save(self.state_dict(), path_saved_ckpt + str(fn) + '.ckpt')
            elif 0.02 < loss <= 0.08:
                fn = int(loss * 100)
                if fn % 2 == 0:
                    torch.save(self.state_dict(), path_saved_ckpt + str(fn) + '.ckpt')
            else:
                torch.save(self.state_dict(), path_saved_ckpt + '.ckpt')


    def predict(self, data=None, n_songs=10, song_idx=0, target_start=None, target_len=None, target_range=None, dirname_post_fix=''):
        datum = np.array(data[song_idx:song_idx+1][0])[None]

        # calculate the index of start of bar6 and the end of bar9
        lbound = np.nonzero(datum[0, :, 1] == 6)[0][0]
        rbound = np.nonzero(datum[0, :, 1] == 9)[0][-1]


        seq_len = datum.shape[1]
        target_full_len = target_range[1] - target_range[0]

        if target_len == None:
            # target_len = np.random.randint(int(target_full_len * args.target_max_percent * 0.75), int(target_full_len * args.target_max_percent))
            # target_len = np.random.randint(int((rbound - lbound) * 0.5), rbound - lbound + 1)
            target_len = rbound - lbound
        if target_start == None:
            # target_start = np.random.randint(target_range[0], int(target_range[0]+ target_full_len * (1-args.target_max_percent)))
            # target_start = np.random.randint(lbound, rbound - target_len + 1)
            target_start = lbound
        target_end = target_start + target_len

        # target_start = np.random.randint(0, int(seq_len - 8))
        # target_len = 8
        # target_end = target_start + target_len

        print("Song idx: %d, song length: %d" % (song_idx, seq_len))
        print("Target_start: %d, target_len: %d" % (target_start, target_len))

        first_onset = datum[0, target_start, [1, 2]]
        target_begin_token = [self.w2e[etype][datum[0, target_start, j]].split(' ')[1] for j, etype in enumerate(self.w2e)]
        target_end_token = [self.w2e[etype][datum[0, target_start+target_len-1, j]].split(' ')[1] for j, etype in enumerate(self.w2e)]
        save_midi_folder = ("song%d_(start)bar%dpos%s_(end)bar%dpos%s" + dirname_post_fix) % \
                            (song_idx, int(target_begin_token[1])+1, target_begin_token[2], int(target_end_token[1])+1, target_end_token[2])
        save_midi_folder = save_midi_folder.replace('/', '|')
        os.makedirs(save_midi_folder, exist_ok=True)
        print("save midi to `%s`" % save_midi_folder)

        # save prime
        d_prime = np.copy(datum[0])
        d_prime[target_start : target_end] = self.mask_word_np
        to_midi_absolute_bar(d_prime, self.w2e, os.path.join(save_midi_folder, "song%d_prime_len%d.midi" % (song_idx, datum.shape[1] - target_len)))

        # absolute bar encoding -> relative bar encoding
        datum[0, 1:, 1] = datum[0, 1:, 1] - datum[0, :-1, 1]
        datum[0, :, 1][datum[0, :, 1] > 1] = 1  # avoid bug when there are empty bars

        # mask out target part
        datum_tmp = np.empty(shape=(1, seq_len+args.mask_len-target_len, len(self.e2w)), dtype=np.long)
        datum_tmp[:, :target_start] = datum[:, :target_start]
        datum_tmp[:, target_start:target_start+args.mask_len] = self.mask_word_np
        datum_tmp[:, target_start + args.mask_len : target_start + args.mask_len + seq_len - target_end] = datum[:, target_end:seq_len]
        datum = datum_tmp


        for sidx in range(n_songs):
            input_ids = torch.from_numpy(datum).to(device)

            attn_mask = None

            y = self.forward(input_ids)

            # sampling
            cnt = 0
            for note_idx in range(target_start, target_start+args.mask_len):
                y_logits = []
                for i, etype in enumerate(self.e2w):
                    y_logits.append(y[i][0, note_idx, :])
                cur_word = []
                for i, etype in enumerate(self.e2w):
                    cur_word.append(self.nucleus(y_logits[i], p=0.9, t=0.8))
                cur_word = np.array(cur_word)
                cnt += 1

                input_ids[:, note_idx] = torch.from_numpy(cur_word)

                stop = False
                for i, etype in enumerate(self.e2w):
                    if 'BLK' in self.w2e[etype][cur_word[i]]:
                        stop = True
                        break
                if stop:
                    break

            input_ids = input_ids.cpu().detach().numpy()[0]

            # for i in range(input_ids.shape[0]):
            #     if i in range(condition_len, input_ids.shape[0]):
            #         print("(target)", end=' ')
            #     else:
            #         print("        ", end=' ')
            #     print(*[self.w2e[etype][input_ids[i, j]] for j, etype in enumerate(self.w2e)], sep=', ')

            # print("\n" + "=" * 80)
            to_midi(input_ids, self.w2e, os.path.join(save_midi_folder, "song%d_%d_len%d.midi" % (song_idx, sidx, seq_len - target_len + cnt)))

        print("=" * 80)

    def nucleus(self, logit, p=0.9, t=1.2):
        logit = logit.cpu().detach().numpy()
        probs = temperature(logits=logit, temperature=t)
        cur_word = nucleus(probs, p=p)
        return cur_word

    def load_finetune_checkpoint(self, path):
        m = torch.load(path)
        model_dict = self.state_dict()
        for k in m.keys():
            if 'distance_embedding' in k or 'position_' in k:
                continue

            if k in model_dict:
                pname = k
                pval = m[k]
                model_dict[pname] = pval.clone().to(model_dict[pname].device)

        self.load_state_dict(model_dict)


if __name__ == '__main__':
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)
    model = BertForPredictingMiddleNotes(configuration, e2w, w2e).to(device)

    if args.train:
        training_data = prepare_data.prepare_data_for_training(args.data_file, is_train=True, e2w=e2w, w2e=w2e, n_step_bars=args.n_step_bars, max_len=args.max_seq_len)
        model.train(training_data=training_data, n_epochs=args.train_epochs)
    else:
        test_data = prepare_data.prepare_data_for_training(args.data_file, is_train=False, e2w=e2w, w2e=w2e, n_step_bars=args.n_step_bars, max_len=10000)
        # model.load_finetune_checkpoint(args.ckpt_path)
        model.load_state_dict(torch.load(args.ckpt_path))
        # model.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device('cpu')))

        target_range = [(0, len(x)) for x in test_data]

        song_idx_step_size = 25
        for i in range(300, len(test_data), song_idx_step_size):
            model.predict(data=test_data, n_songs=3, song_idx=i, target_range=target_range[i])

