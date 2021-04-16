import numpy as np
import math
import sys
import time
import datetime
import os
import copy

from transformers import XLNetTokenizer, XLNetModel, XLNetConfig, AdamW

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import prepare_data
import pickle
import argparse

np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description='')

# training setup
parser.add_argument('--dict-file', type=str, default='/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/predict-middle-notes/dictionary.pickle')
parser.add_argument('--data-file', type=str, default='/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/predict-middle-notes/worded_data.pickle')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--save-path', type=str, default="/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/predict-middle-notes/trained-models/partial-target-test")
parser.add_argument('--batch-size', type=int, default=6)
parser.add_argument('--target-max-percent', type=float, default=0.2, help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
parser.add_argument('--n-step-bars', type=int, default=16, help='how many bars to step before next training data fetching (the smaller the more training data)')
parser.add_argument('--max-seq-len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
parser.add_argument('--train-epochs', type=int, default=2000, help='number of training epochs')
parser.add_argument('--init-lr', type=float, default=1e-4, help='initial learning rate')

# for prediction phase
parser.add_argument('--test-data-file', type=str, default='/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/predict-middle-notes-test/worded_data.pickle')
parser.add_argument('--ckpt-path', type=str, default="/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/predict-middle-notes/trained-models/short-target-mapping/loss34.ckpt")
parser.add_argument('--song-idx', type=int, default=170)

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

configuration = XLNetConfig().from_dict({
  "_name_or_path": "xlnet-predict-middle-notes",
  "architectures": [
    "XLNetLMHeadModel"
  ],
  "attn_type": "bi",
  "bi_data": False,
  "bos_token_id": 10000,
  "clamp_len": -1,
  # "d_head": 64,
  "d_inner": 3072,
  "d_model": 768,
  "dropout": 0.1,
  "end_n_top": 5,
  "eos_token_id": 2,
  "ff_activation": "gelu",
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-12,
  "mem_len": None, # null
  "model_type": "xlnet",
  "n_head": 8,  # 12 originally
  "n_layer": 12,
  "pad_token_id": 10000,
  "reuse_len": None, # null,
  "same_length": False,
  "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_dropout": 0.1,
  "summary_type": "last",
  "summary_use_proj": True,
  "untie_r": True,
  "use_mems_eval": True,
  "use_mems_train": True,
  # "vocab_size": 32000
})

# --- write tool --- #
def to_midi(data, word2event, path_outfile):
    tes = []    # tuple events
    for e in data:
        e = [word2event[etype][e[i]] for i, etype in enumerate(word2event)]
        te = prepare_data.GroupEvent(Tempo=int(e[0].split(' ')[1]),
                                     Bar=int(e[1].split(' ')[1]),
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
        last_index = np.where(after_threshold)[0][1]

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

class XLNetForPredictingMiddleNotes(torch.nn.Module):
    def __init__(self, xlnet_config, e2w, w2e, is_train=None):
        super(XLNetForPredictingMiddleNotes, self).__init__()
        self.xlnet = XLNetModel(xlnet_config, is_train=is_train)
        self.xlnet_config = xlnet_config
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        # token types: [Tempo, Bar, Position, Pitch, Duration, Velocity]
        self.n_tokens = []
        for key in e2w:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.tempo_pad_word = self.e2w['Tempo']['Tempo <PAD>']

        self.eos_word = torch.Tensor([self.e2w[etype]['%s <EOS>' % etype] for etype in self.e2w]).long().to(device)
        self.bos_word = torch.Tensor([self.e2w[etype]['%s <BOS>' % etype] for etype in self.e2w]).long().to(device)

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.e2w):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types to feed into xlnet
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), xlnet_config.d_model)

        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(self.e2w):
            self.proj.append(nn.Linear(xlnet_config.d_model, self.n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)


    def forward(self, input_ids, attention_mask, perm_mask, target_mapping, input_ids_g=None):
        """
        Args:
            input_ids: of shape [bsz, seq_len, n_event_type]. Input for content stream.
        """
        # convert input_ids into embeddings and merge them through linear layer
        embs =[]
        for i, key in enumerate(self.e2w):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        # (for query stream) convert input_ids into embeddings and merge them through linear layer
        embs_g =[]
        for i, key in enumerate(self.e2w):
            embs_g.append(self.word_emb[i](input_ids_g[..., i]))
        embs_g = torch.cat([*embs_g], dim=-1)
        emb_linear_g = self.in_linear(embs_g)

        # feed to xlnet
        y = self.xlnet(inputs_embeds=emb_linear,
                       attention_mask=attention_mask,
                       perm_mask=perm_mask,
                       target_mapping=target_mapping,
                       inputs_embeds_g=emb_linear_g,
                       bar_ids=input_ids[..., 1])
        y = y.last_hidden_state

        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))

        return ys


    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def train(self, training_data=None, n_epochs=None):
        os.makedirs(args.save_path, exist_ok=True)
        path_saved_ckpt = os.path.join(args.save_path, 'loss')

        start_time = time.time()
        optimizer = AdamW(self.parameters(), lr=args.init_lr, weight_decay=0.01)
        num_batches = len(training_data) // args.batch_size
        for epoch in range(args.train_epochs):
            total_losses = 0
            for train_iter in range(num_batches):
                input_ids = torch.from_numpy(training_data[train_iter*args.batch_size:(train_iter+1)*args.batch_size]).to(device)

                # attn_mask: mask to avoid attending to <PAD> tokens
                # 0: do not attend, 1: attend
                attn_mask = (input_ids[:, :, 0] != self.tempo_pad_word).float()

                # decide the range to be predicted: `target_start` to `target_start + target_len`
                valid_seq_lengths = [torch.nonzero(seq)[-1][0] + 1 for seq in attn_mask] # seq length without <PAD> tokens
                target_starts = [np.random.randint(int(seq_len * (1 - args.target_max_percent))) for seq_len in valid_seq_lengths]
                target_lens = [np.random.randint(int(seq_len * args.target_max_percent / 2), int(seq_len * args.target_max_percent)) for seq_len in valid_seq_lengths]

                # prepare input_ids_g: use bar+pos instead of sin+cos embeddings as position information
                input_ids_g = torch.zeros(args.batch_size, max(target_lens), len(self.e2w)).long().to(device)
                for b in range(args.batch_size):
                    input_ids_g[b, :target_lens[b]] = input_ids[b, target_starts[b]:target_starts[b]+target_lens[b]]
                    input_ids_g[b, :target_lens[b], [0, 3, 4, 5]] = self.bos_word[[0, 3, 4, 5]]  # mask out tokens except bar & pos

                # generate perm_mask
                # 0: attend, 1: do not attend
                perm_mask = torch.ones(args.batch_size, args.max_seq_len, args.max_seq_len).to(device)
                for b in range(args.batch_size):
                    perm_mask[b, :, :target_starts[b]] = 0
                    perm_mask[b, :, target_starts[b] + target_lens[b]:valid_seq_lengths[b]] = 0
                    for i in range(target_starts[b], target_starts[b]+target_lens[b]):
                        perm_mask[b, i, target_starts[b]:i] = 0

                # target mapping: partial prediction
                target_mapping = torch.zeros(args.batch_size, max(target_lens), args.max_seq_len).to(device)
                for b in range(args.batch_size):
                    for i, j in enumerate(range(target_starts[b], target_starts[b]+target_lens[b])):
                        target_mapping[b, i, j] = 1

                y = self.forward(input_ids, attn_mask, perm_mask, target_mapping, input_ids_g=input_ids_g)

                # reshape (b, s, f) -> (b, f, s)
                for i, etype in enumerate(self.e2w):
                    y[i] = y[i][:, ...].permute(0, 2, 1)


                # calculate losses
                target = torch.zeros(args.batch_size, max(target_lens), len(self.e2w), dtype=torch.long).to(device)
                loss_mask = torch.zeros(args.batch_size, max(target_lens))
                for b in range(args.batch_size):
                    target[b, :target_lens[b], [0, 3, 4, 5]] = input_ids[b, target_starts[b]:target_starts[b]+target_lens[b], [0, 3, 4, 5]]

                    # next onset prediction
                    target[b, :target_lens[b]-1, [1, 2]] = input_ids[b, target_starts[b]+1:target_starts[b]+target_lens[b], [1, 2]]
                    target[b, target_lens[b]-1, [1, 2]] = self.eos_word[[1, 2]]

                    loss_mask[b, :target_lens[b]] = 1
                losses = []
                for i, etype in enumerate(self.e2w):
                    losses.append(self.compute_loss(y[i], target[..., i].to(device), loss_mask.to(device)))
                total_loss = sum(losses) / len(self.e2w)

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


    def predict(self, data=None, n_songs=10, song_idx=0, target_start=None, target_len=None):
        datum = np.array(data[song_idx:song_idx+1][0])[None]
        seq_len = datum.shape[1]

        if target_start == None:
            target_start = np.random.randint(int(seq_len * (1 - args.target_max_percent)))
        if target_len == None:
            target_len = np.random.randint(int(seq_len * args.target_max_percent * 0.75), int(seq_len * args.target_max_percent))

        print("Song idx: %d, song length: %d" % (song_idx, seq_len))
        print("Target_start: %d, target_len: %d" % (target_start, target_len))

        first_onset = datum[0, target_start, [1, 2]]
        target_begin_token = [self.w2e[etype][datum[0, target_start, j]].split(' ')[1] for j, etype in enumerate(self.w2e)]
        target_end_token = [self.w2e[etype][datum[0, target_start+target_len-1, j]].split(' ')[1] for j, etype in enumerate(self.w2e)]
        save_midi_folder = "song%d_(start)bar%dpos%s_(end)bar%dpos%s" % (song_idx, int(target_begin_token[1])+1, target_begin_token[2], int(target_end_token[1])+1, target_end_token[2])
        save_midi_folder = save_midi_folder.replace('/', '|')
        os.makedirs(save_midi_folder, exist_ok=True)
        print("save midi to `%s`" % save_midi_folder)

        # A_C -> AC
        datum[:, target_start:seq_len-target_len] = datum[:, target_start+target_len:]
        datum = datum[:, :seq_len-target_len]

        for sidx in range(n_songs):
            input_ids = torch.from_numpy(datum).to(device)
            next_onset = torch.from_numpy(first_onset).long().to(device)

            condition_len = input_ids.shape[1]

            attn_mask = None

            while True:
                input_ids = torch.cat([input_ids, self.bos_word[None, None]], dim=1)
                input_ids_g = torch.clone(self.bos_word)
                input_ids_g[[1, 2]] = next_onset
                input_ids_g = input_ids_g[None, None]

                # generate perm_mask
                # 0: attend, 1: do not attend
                perm_mask = torch.ones(1, input_ids.shape[1], input_ids.shape[1]).to(device)
                perm_mask[0, :, :condition_len] = 0
                for i in range(condition_len, input_ids.shape[1]):
                    perm_mask[0, i, condition_len:i] = 0

                # target mapping: partial prediction
                target_mapping = torch.zeros(1, 1, input_ids.shape[1]).to(device)
                target_mapping[0, 0, -1] = 1

                y = self.forward(input_ids, attn_mask, perm_mask, target_mapping, input_ids_g=input_ids_g)

                # sampling
                y_logits = []
                for i, etype in enumerate(self.e2w):
                    y_logits.append(y[i][0, -1, :])
                cur_word = []
                for i, etype in enumerate(self.e2w):
                    cur_word.append(self.nucleus(y_logits[i], p=0.9, t=0.8))
                cur_word = np.array(cur_word)

                input_ids[0, -1, [1, 2]] = next_onset
                input_ids[0, -1, [0, 3, 4, 5]] = torch.from_numpy(cur_word).to(device)[[0, 3, 4, 5]]
                next_onset = torch.from_numpy(cur_word).to(device)[[1, 2]]

                if 'EOS' in self.w2e['Bar'][cur_word[1]]:
                    break
                if 'EOS' in self.w2e['Position'][cur_word[2]]:
                    break
                if input_ids.shape[1] >= 1000:
                    break

            input_ids = input_ids.cpu().detach().numpy()[0]

            # for i in range(input_ids.shape[0]):
            #     if i in range(condition_len, input_ids.shape[0]):
            #         print("(target)", end=' ')
            #     else:
            #         print("        ", end=' ')
            #     print(*[self.w2e[etype][input_ids[i, j]] for j, etype in enumerate(self.w2e)], sep=', ')

            to_midi(input_ids, self.w2e, os.path.join(save_midi_folder, "song%d_%d.midi" % (song_idx, sidx)))

        print("=" * 80)

    def nucleus(self, logit, p=0.9, t=1.2):
        logit = logit.cpu().detach().numpy()
        probs = temperature(logits=logit, temperature=t)
        cur_word = nucleus(probs, p=p)
        return cur_word



if __name__ == '__main__':
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    if args.train:
        model = XLNetForPredictingMiddleNotes(configuration, e2w, w2e, is_train=True).to(device)

        # a = torch.Tensor([[0, 0, 0, 1, 1, 2], [0, 0, 1, 2, 2, 2]])
        # model.xlnet.relative_positional_encoding(bar_ids=a)
        # sys.exit(0)

        training_data = prepare_data.prepare_data_for_training(args.data_file, is_train=True, e2w=e2w, w2e=w2e, n_step_bars=args.n_step_bars, max_len=args.max_seq_len)
        model.train(training_data=training_data, n_epochs=args.train_epochs)
    else:
        model = XLNetForPredictingMiddleNotes(configuration, e2w, w2e, is_train=False).to(device)
        test_data = prepare_data.prepare_data_for_training(args.data_file, is_train=False, e2w=e2w, w2e=w2e, n_step_bars=args.n_step_bars, max_len=args.max_seq_len)
        model.load_state_dict(torch.load(args.ckpt_path))
        for i in range(0, len(test_data), 10):
            model.predict(data=test_data, n_songs=10, song_idx=i)

