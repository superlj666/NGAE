from __future__ import print_function
import os
import sys
import math
import pickle
import pdb
import argparse
import random
from tqdm import tqdm
from shutil import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy.io
from scipy.linalg import qr 
import igraph
from random import shuffle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util import *
from models import *
from bayesian_optimization.evaluate_BN import Eval_BN
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-type', default='ENAS',
                    help='ENAS: ENAS-format CNN structures; BN: Bayesian networks')
parser.add_argument('--data-name', default='final_structures6', help='graph dataset name')
parser.add_argument('--nvt', type=int, default=6, help='number of different node types, \
                    6 for final_structures6, 8 for asia_200k')
parser.add_argument('--save-appendix', default='_NGAE_sigmoid', 
                    help='what to append to data-name as save-name for results')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--sample-number', type=int, default=20, metavar='N',
                    help='how many samples to generate each time')
parser.add_argument('--no-test', action='store_true', default=False,
                    help='if True, merge test with train, i.e., no held-out set')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--keep-old', action='store_true', default=True,
                    help='if True, do not remove any old data in the result folder')
parser.add_argument('--only-test', action='store_true', default=True,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--only-search', action='store_true', default=False,
                    help='if True, perform search on latent space')
parser.add_argument('--search-strategy', default='random',
                    help='search strategy, including random and optimal')
parser.add_argument('--search-optimizer', default='Newton',
                    help='optimizer, including sgd and Newton')
parser.add_argument('--small-train', action='store_true', default=False,
                    help='if True, use a smaller version of train set')
# model settings
parser.add_argument('--model', default='NGAE', help='model to use: NGAE, SVAE, \
                    NGAE_fast, NGAE_BN, SVAE_oneshot, NGAE_GCN')
parser.add_argument('--load-latest-model', action='store_true', default=False,
                    help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=300, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--hs', type=int, default=501, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=56, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=True,
                    help='whether to use bidirectional encoding')
parser.add_argument('--predictor', action='store_true', default=True,
                    help='whether to train a performance predictor from latent\
                    encodings and a VAE at the same time')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=128, metavar='N',
                    help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--all-gpus', action='store_true', default=False,
                    help='use all available GPUs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--search-samples', type=int, default=10, metavar='N',
                    help='the number of samples for searching')
parser.add_argument('--train-from-scratch', action='store_true', default=False,
                    help='if True, perform train on selected architectures')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
random.seed(args.seed)
print(args)
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}{}'.format(args.data_name, args.save_appendix))
if args.predictor:
    args.res_dir += '{}'.format('_predictor')

pkl_name = './results/final_structures6_NGAE_sigmoid_predictor/final_structures6.pkl'
with open(pkl_name, 'rb') as f:
    train_data, test_data, graph_args = pickle.load(f)

model = eval(args.model)(
        graph_args.max_n, 
        graph_args.num_vertex_type, 
        graph_args.START_TYPE, 
        graph_args.END_TYPE, 
        hs=args.hs, 
        nz=args.nz, 
        bidirectional=args.bidirectional
        )
if args.predictor:
    predictor = nn.Sequential(
            nn.Linear(args.nz, args.hs), 
            nn.Tanh(), 
            nn.Linear(args.hs, 1),
            nn.Sigmoid()
            )
    model.predictor = predictor
    model.mseloss = nn.MSELoss(reduction='sum')
# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
model.to(device)

epoch = 300
load_module_state(model, os.path.join(args.res_dir, 
                                      'model_checkpoint{}.pth'.format(epoch)))
load_module_state(optimizer, os.path.join(args.res_dir, 
                                          'optimizer_checkpoint{}.pth'.format(epoch)))
load_module_state(scheduler, os.path.join(args.res_dir, 
                                          'scheduler_checkpoint{}.pth'.format(epoch)))
test_data = test_data[200:300]
y_pred_list = []
y_true_list = []
for (g, y) in test_data:
    mu, logvar = model.encode(g)
    y_pred = model.predictor(mu)
    y_true_list.append(y)
    y_pred_list.append(y_pred[0][0].cpu().detach().numpy().item())

fig = plt.figure()
scat = plt.scatter(np.array(y_true_list), np.array(y_pred_list), label='sampled point') #  c = 'r', marker= 'o'
x = y_true_list
y = y_pred_list
parameter = np.polyfit(x, y, 1)
f = np.poly1d(parameter)
ln = plt.plot(x, f(x), color='r', label='fit line')
plt.xlabel('True accuracy')
plt.ylabel('Predict accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('experiments/exp2_scatter.pdf')