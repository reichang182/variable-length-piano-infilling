from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
import torch
import prepare_data
import pickle
import argparse

parser = argparse.ArgumentParser(description='')

# training setup
parser.add_argument('--dict-file', type=str, default='/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/predict-middle-notes/dictionary.pickle')
parser.add_argument('--data-file', type=str, default='/home/csc63182/NAS-189/homes/csc63182/data/remi-1700/predict-middle-notes/worded_data.pickle')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--save-path', type=str, default="/home/csc63182/NAS-189/homes/csc63182/remi/checkpoints/Linformer-6tuple-bs28")
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--n-step-bars', type=int, default=16, help='how many bars to step before next training data fetching (the smaller the more training data)')

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

class XLNetForPredictingMiddleNotes(torch.nn.Module):
    def __init__(self, xlnet_config):
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.xlnet = XLNetModel(xlnet_config)
        self.emb_sizes = [256, 256, 256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

# model = XLNetModel.from_pretrained('xlnet-base-cased')
# configuration = model.config
# print(configuration)
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state
#
# print(last_hidden_states.shape)

with open(args.dict_file, 'rb') as f:
    e2w, w2e = pickle.load(f)
train_data = prepare_data.prepare_data_for_training(args.data_file, is_train=args.train, e2w=e2w, w2e=w2e, n_step_bars=args.n_step_bars)
