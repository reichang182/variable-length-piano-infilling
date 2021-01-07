import numpy as np
import math
import sys
import time
import datetime
import os

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
parser.add_argument('--save-path', type=str, default="/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/predict-middle-notes/trained-models/partial-target")
parser.add_argument('--batch-size', type=int, default=6)
parser.add_argument('--target-max-percent', type=float, default=0.2, help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
parser.add_argument('--n-step-bars', type=int, default=16, help='how many bars to step before next training data fetching (the smaller the more training data)')
parser.add_argument('--max-seq-len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
parser.add_argument('--train-epochs', type=int, default=1000, help='number of training epochs')
parser.add_argument('--init-lr', type=float, default=1e-4, help='initial learning rate')

# for testing phase
parser.add_argument('--ckpt-path', type=str, default="/home/csc63182/NAS-189/homes/csc63182/remi/checkpoints/Linformer-6tuple-bs28/saved_10.ckpt")

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

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class XLNetForPredictingMiddleNotes(torch.nn.Module):
    def __init__(self, xlnet_config, e2w, w2e):
        super(XLNetForPredictingMiddleNotes, self).__init__()
        self.xlnet = XLNetModel(xlnet_config)
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

    def forward(self, input_ids, attention_mask, perm_mask):
        # convert input_ids into embeddings and merge them through linear layer
        embs =[]
        for i, key in enumerate(self.e2w):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        # feed to xlnet
        y = self.xlnet(inputs_embeds=emb_linear, attention_mask=attention_mask, perm_mask=perm_mask)
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

                # generate perm_mask
                # 0: attend, 1: do not attend
                perm_mask = torch.ones(args.batch_size, args.max_seq_len, args.max_seq_len).to(device)
                for b in range(args.batch_size):
                    perm_mask[b, :, :target_starts[b]] = 0
                    perm_mask[b, :, target_starts[b] + target_lens[b]:valid_seq_lengths[b]] = 0
                    for i in range(target_starts[b], target_starts[b]+target_lens[b]):
                        perm_mask[b, i, target_starts[b]:i] = 0

                y = self.forward(input_ids, attn_mask, perm_mask)

                # reshape (b, s, f) -> (b, f, s)
                for i, etype in enumerate(self.e2w):
                    y[i] = y[i][:, ...].permute(0, 2, 1)


                # calculate losses
                loss_mask = torch.zeros(args.batch_size, args.max_seq_len)
                for b in range(args.batch_size):
                    loss_mask[b, target_starts[b]:target_starts[b]+target_lens[b]] = 1
                losses = []
                for i, etype in enumerate(self.e2w):
                    losses.append(self.compute_loss(y[i], input_ids[..., i], loss_mask.to(device)))
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


if __name__ == '__main__':
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)
    model = XLNetForPredictingMiddleNotes(configuration, e2w, w2e).to(device)
    training_data = prepare_data.prepare_data_for_training(args.data_file, is_train=True, e2w=e2w, w2e=w2e, n_step_bars=args.n_step_bars, max_len=args.max_seq_len)
    model.train(training_data=training_data, n_epochs=args.train_epochs)

