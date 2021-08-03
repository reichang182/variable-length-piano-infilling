import numpy as np
import math
import sys
import time
import datetime
import os
import copy

from transformers import TransfoXLConfig, TransfoXLModel, AdamW

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import prepare_data
import pickle
import argparse

import copy

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description='')

# training setup
parser.add_argument('--dict-file', type=str, default='../../dictionary.pickle')
parser.add_argument('--data-file', type=str, default='../../worded_data.pickle')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--save-path', type=str, default="trained-model")
parser.add_argument('--batch-size', type=int, default=6)
parser.add_argument('--target-max-percent', type=float, default=0.2, help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
parser.add_argument('--n-step-bars', type=int, default=8, help='how many bars to step before next training data fetching (the smaller the more training data)')
parser.add_argument('--max-seq-len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
parser.add_argument('--train-epochs', type=int, default=2000, help='number of training epochs')
parser.add_argument('--init-lr', type=float, default=1e-4, help='initial learning rate')

# for prediction phase
parser.add_argument('--test-data-file', type=str, default='../../worded_data.pickle')
parser.add_argument('--ckpt-path', type=str, default="trained-model/loss.ckpt")
parser.add_argument('--song-idx', type=int, default=170)

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

configuration = TransfoXLConfig().from_dict({
  "_name_or_path": "ILM-predict-middle-notes",
  # "architectures": [
  #   "XLNetLMHeadModel"
  # ],
  # "attn_type": "bi",
  # "bi_data": False,
  # "bos_token_id": 10000,
  # "clamp_len": -1,
  # "d_head": 64,
  "d_inner": 3072,
  "d_model": 768,
  "dropout": 0.1,
  # "end_n_top": 5,
  # "eos_token_id": 2,
  # "ff_activation": "gelu",
  # "initializer_range": 0.02,
  # "layer_norm_eps": 1e-12,
  # "mem_len": 512, # null
  # "model_type": "xlnet",
  "n_head": 8,  # 12 originally
  "n_layer": 12,
  # "pad_token_id": 10000,
  "reuse_len": None, # null,
  "same_length": False,
  # "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_dropout": 0.1,
  "summary_type": "last",
  "summary_use_proj": True,
  "untie_r": True,
  "use_mems_eval": True,
  "use_mems_train": True,
  # "vocab_size": 32000
})

def show_events(data, word2event, target=None, loss_mask=None):
    print("\n\n" + "=" * 80)
    tes = []    # tuple events
    for seq_idx, (e, target) in enumerate(zip(data, target)):
        e = [word2event[etype][e[i]] for i, etype in enumerate(word2event)]
        target = [word2event[etype][target[i]] for i, etype in enumerate(word2event)]
        print(e)
        print(target, end='  ')

        if loss_mask != None and loss_mask[seq_idx] == 1:
            print('[target]\n')
        else:
            print('\n')

# --- write tool --- #
def to_midi_prime(data, word2event, path_outfile):
    tes = []    # tuple events
    # print("\n" + "=" * 80)
    for e in data:
        e_word = copy.deepcopy(e)
        e = [word2event[etype][e[i]] for i, etype in enumerate(word2event)]
        # print(e)
        skip = False
        for i, etype in enumerate(word2event):
            if 'BLK' in e[i] or 'SEP' in e[i] or 'ANS' in e[i]:
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
    # print("\n" + "=" * 80)
    cur_bar = 0
    for e in data:
        e_word = copy.deepcopy(e)
        e = [word2event[etype][e[i]] for i, etype in enumerate(word2event)]
        # print(e)
        skip = False
        for i, etype in enumerate(word2event):
            if 'BLK' in e[i] or 'SEP' in e[i] or 'ANS' in e[i]:
                skip = True
        if skip:
            continue

        if e_word[1] == 1: # new bar event
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

# ILM model: similar approach to "Enabling Language Models to Fill in the Blanks"
class TransformerXLForPredictingMiddleNotes(torch.nn.Module):
    def __init__(self, transfoXLConfig, e2w, w2e):
        super(TransformerXLForPredictingMiddleNotes, self).__init__()
        self.transfoXL = TransfoXLModel(transfoXLConfig)
        self.transfoXLConfig = transfoXLConfig
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
        self.sep_word_np = np.array([self.e2w[etype]['%s <SEP>' % etype] for etype in self.e2w], dtype=np.long)
        self.ans_word_np = np.array([self.e2w[etype]['%s <ANS>' % etype] for etype in self.e2w], dtype=np.long)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.e2w], dtype=np.long)

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.e2w):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types to feed into transformer-XL
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), transfoXLConfig.d_model)

        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(self.e2w):
            self.proj.append(nn.Linear(transfoXLConfig.d_model, self.n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)


    def forward(self, input_ids, mems=None):
        # convert input_ids into embeddings and merge them through linear layer
        embs =[]
        for i, key in enumerate(self.e2w):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        # feed to transformer-XL
        y = self.transfoXL(inputs_embeds=emb_linear, mems=mems)
        new_mems = y.mems
        y = y.last_hidden_state

        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))

        return ys, new_mems


    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

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
            for train_iter in range(num_batches):
                # input_ids = torch.from_numpy(training_data[train_iter*args.batch_size:(train_iter+1)*args.batch_size]).to(device)
                ori_seq_batch = training_data[train_iter*args.batch_size:(train_iter+1)*args.batch_size]
                start_end_batch = start_end[train_iter * args.batch_size : (train_iter + 1) * args.batch_size]

                # decide the range to be predicted: `target_start` to `target_start + target_len`
                # target_starts = [np.random.randint(0, int(len(seq) * (1 - args.target_max_percent))) for seq in ori_seq_batch]
                # target_lens = [np.random.randint(int(len(seq) * args.target_max_percent / 2), int(len(seq) * args.target_max_percent))
                #                for seq in ori_seq_batch]
                target_lens = [np.random.randint(int((end - start) * 0.5), end - start + 1) for (start, end) in start_end_batch]
                target_starts = [np.random.randint(start, end - target_len + 1) for (start, end), target_len in zip(start_end_batch, target_lens)]
                target_ends = [s + l for s, l in zip(target_starts, target_lens)]

                # prepare input for training
                # [x] to [~x <blk> ~x <sep> x_ans <ans> <pad> <pad> ...]
                input_ids = np.full((args.batch_size, args.max_seq_len, len(self.e2w)), self.pad_word_np)
                for b in range(args.batch_size):
                    if target_starts[b] != 0:
                        input_ids[b, 0:target_starts[b]] = ori_seq_batch[b][0:target_starts[b]] # pre-blank ~x
                    input_ids[b, target_starts[b]] = self.blk_word_np   # blank
                    input_ids[b, target_starts[b]+1:len(ori_seq_batch[b])-target_lens[b]+1] = ori_seq_batch[b][target_ends[b]:]    # post-blank ~x
                    input_ids[b, len(ori_seq_batch[b])-target_lens[b]+1] = self.sep_word_np    # sep
                    input_ids[b, len(ori_seq_batch[b])-target_lens[b]+2:len(ori_seq_batch[b])+2] = ori_seq_batch[b][target_starts[b]:target_ends[b]] # target
                    input_ids[b, len(ori_seq_batch[b])+2] = self.ans_word_np    # ans

                # print("=" * 80)
                input_ids = torch.from_numpy(input_ids).to(device)

                y, _ = self.forward(input_ids)

                # get the most likely choice with max
                # outputs = []
                # for i, etype in enumerate(self.e2w):
                #     output = np.argmax(y[i].cpu().detach().numpy(), axis=-1)
                #     outputs.append(output)
                # outputs = np.stack(outputs, axis=-1)
                # print("shape of outputs:", outputs.shape)

                # reshape (b, s, f) -> (b, f, s)
                for i, etype in enumerate(self.e2w):
                    y[i] = y[i][:, ...].permute(0, 2, 1)


                # calculate losses
                # target = torch.clone(input_ids).detach()
                target = torch.zeros_like(input_ids).long()
                target[:, :-1] = input_ids[:, 1:]
                loss_mask = torch.zeros(args.batch_size, args.max_seq_len)
                for b in range(args.batch_size):
                    loss_mask[b, len(ori_seq_batch[b])-target_lens[b]+1:len(ori_seq_batch[b])+2] = 1
                losses = []
                for i, etype in enumerate(self.e2w):
                    losses.append(self.compute_loss(y[i], target[..., i].to(device), loss_mask.to(device)))
                total_loss = sum(losses) / len(self.e2w)

                # show_events(input_ids[1].cpu().numpy(), self.w2e, target=outputs[1], loss_mask=loss_mask[1])

                # udpate
                self.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.parameters(), 3.0)
                optimizer.step()


                # acc
                sys.stdout.write('{}/{} | Loss: {:06f} | {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
                    train_iter, num_batches, total_loss, *losses))
                losses = list(map(float, losses))
                total_losses += total_loss.item()

            runtime = time.time() - start_time
            print('epoch: {}/{} | Loss: {} | time: {}'.format(
                epoch, n_epochs, total_losses/num_batches, str(datetime.timedelta(seconds=runtime))))
            print('    > loss: {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}'.format(*losses))


            loss =  total_losses/num_batches
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

        # calculate the start of bar6 and the end of the bar9
        lbound = np.nonzero(datum[0, :, 1] == 6)[0][0]
        rbound = np.nonzero(datum[0, :, 1] == 9)[0][-1]


        seq_len = datum.shape[1]
        target_full_len = target_range[1] - target_range[0]

        if target_len == None:
            # target_len = np.random.randint(int(target_full_len * args.target_max_percent * 0.5), int(target_full_len * args.target_max_percent * 1.0))
            # target_len = np.random.randint(int((rbound - lbound) * 0.5), rbound - lbound + 1)
            target_len = rbound - lbound
        if target_start == None:
            # target_start = np.random.randint(target_range[0], int(target_range[0]+ target_full_len * (1-args.target_max_percent)))
            # target_start = np.random.randint(lbound, rbound - target_len + 1)
            target_start = lbound

        print("Song idx: %d, song length: %d" % (song_idx, seq_len))
        print("Target_start: %d, target_len: %d" % (target_start, target_len))

        first_onset = datum[0, target_start, [1, 2]]
        target_begin_token = [self.w2e[etype][datum[0, target_start, j]].split(' ')[1] for j, etype in enumerate(self.w2e)]
        target_end_token = [self.w2e[etype][datum[0, target_start+target_len-1, j]].split(' ')[1] for j, etype in enumerate(self.w2e)]
        # save_midi_folder = ("song%d_(start)bar%dpos%s_(end)bar%dpos%s" + dirname_post_fix) % \
        #                     (song_idx, int(target_begin_token[1])+1, target_begin_token[2], int(target_end_token[1])+1, target_end_token[2])
        save_midi_folder = ("song%d" + dirname_post_fix) % (song_idx)
        save_midi_folder = save_midi_folder.replace('/', '|')
        os.makedirs(save_midi_folder, exist_ok=True)
        print("save midi to `%s`" % save_midi_folder)

        # save prime
        d_tmp = np.copy(datum)
        d_tmp[0, target_start : target_start + target_len] = self.blk_word_np
        to_midi_prime(d_tmp[0], self.w2e, os.path.join(save_midi_folder, "song%d_prime_len%d.midi" % (song_idx, datum.shape[1] - target_len)))

        # absolute bar encodings -> relative bar encodings
        datum[0, 1:, 1] = datum[0, 1:, 1] - datum[0, :-1, 1]
        datum[0, :, 1][datum[0, :, 1] > 1] = 1  # avoid bug when there are empty bars

        # A_C -> AC
        datum[:, target_start] = self.blk_word_np
        datum[:, target_start+1:seq_len-target_len+1] = datum[:, target_start+target_len:]
        datum[:, seq_len-target_len+1] = self.sep_word_np
        datum = datum[:, :seq_len-target_len+2]


        for sidx in range(n_songs):
            input_ids = torch.from_numpy(datum).to(device)

            condition_len = input_ids.shape[1]

            attn_mask = None
            mems = None
            first_predict = True

            while True:
                y, mems = self.forward(input_ids[:, :, :])
                # if first_predict:
                #     y, mems = self.forward(input_ids[:, :, :], mems=mems)
                #     first_predict = False
                # else:
                #     y, mems = self.forward(input_ids[:, -1:, :], mems=mems)

                # sampling
                y_logits = []
                for i, etype in enumerate(self.e2w):
                    y_logits.append(y[i][0, -1, :])
                cur_word = []
                for i, etype in enumerate(self.e2w):
                    cur_word.append(self.nucleus(y_logits[i], p=0.9, t=0.8))
                cur_word = np.array(cur_word)

                input_ids = torch.cat([input_ids, torch.from_numpy(cur_word[None, None, :]).to(device)], dim=1)

                stop_gen = False
                for i, etype in enumerate(self.e2w):
                    if 'ANS' in self.w2e[etype][cur_word[i]]:
                        stop_gen = True

                if stop_gen or input_ids.shape[1] >= 450:
                    break

            input_ids = input_ids.cpu().detach().numpy()[0]

            # for i in range(input_ids.shape[0]):
            #     if i in range(condition_len, input_ids.shape[0]):
            #         print("(target)", end=' ')
            #     else:
            #         print("        ", end=' ')
            #     print(*[self.w2e[etype][input_ids[i, j]] for j, etype in enumerate(self.w2e)], sep=', ')
            reordered_input_ids = np.concatenate([input_ids[:target_start], input_ids[seq_len-target_len+1:], input_ids[target_start+1:seq_len-target_len+2]], axis=0)
            to_midi(reordered_input_ids, self.w2e, os.path.join(save_midi_folder, "song%d_%d_len%d.midi" % (song_idx, sidx, input_ids.shape[0])))

        print("=" * 80)

    def nucleus(self, logit, p=0.9, t=1.2):
        logit = logit.cpu().detach().numpy()
        probs = temperature(logits=logit, temperature=t)
        cur_word = nucleus(probs, p=p)
        return cur_word



if __name__ == '__main__':
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)
    model = TransformerXLForPredictingMiddleNotes(configuration, e2w, w2e).to(device)

    if args.train:
        training_data = prepare_data.prepare_data_for_training(args.data_file, is_train=True, e2w=e2w, w2e=w2e, n_step_bars=args.n_step_bars, max_len=args.max_seq_len)
        model.load_state_dict(torch.load(args.ckpt_path))
        model.train(training_data=training_data, n_epochs=args.train_epochs)
    else:
        test_data = prepare_data.prepare_data_for_training(args.data_file, is_train=False, e2w=e2w, w2e=w2e, n_step_bars=args.n_step_bars, max_len=10000)
        model.load_state_dict(torch.load(args.ckpt_path))

        target_range = [(0, len(x)) for x in test_data]

        song_idx_step_size = 25
        for i in range(300, len(test_data), song_idx_step_size):
            model.predict(data=test_data, n_songs=3, song_idx=i, target_range=target_range[i])

