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
from thop.profile import profile


parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-type', default='ENAS',
                    help='ENAS: ENAS-format CNN structures; BN: Bayesian networks')
parser.add_argument('--data-name', default='final_structures6', help='graph dataset name')
parser.add_argument('--nvt', type=int, default=6, help='number of different node types, \
                    6 for final_structures6, 8 for asia_200k')
parser.add_argument('--save-appendix', default='_NAOGE_debug', 
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
parser.add_argument('--only-test', action='store_true', default=False,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--only-search', action='store_true', default=False,
                    help='if True, perform search on latent space')
parser.add_argument('--search-strategy', default='random',
                    help='search strategy, including random and optimal')
parser.add_argument('--search-optimizer', default='sgd',
                    help='optimizer, including sgd and Newton')
parser.add_argument('--small-train', action='store_true', default=False,
                    help='if True, use a smaller version of train set')
# model settings
parser.add_argument('--model', default='NGAE', help='model to use: NGAE, \
                    NGAE_fast, NGAE_BN, NGAE_GCN')
parser.add_argument('--load-latest-model', action='store_true', default=False,
                    help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--hs', type=int, default=501, metavar='N', # 501
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=56, metavar='N',  # 56
                    help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=True,
                    help='whether to use bidirectional encoding')
parser.add_argument('--predictor', action='store_true', default=False,
                    help='whether to train a performance predictor from latent\
                    encodings and a VAE at the same time')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=64, metavar='N',
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

'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}{}'.format(args.data_name, args.save_appendix))
if args.predictor:
    args.res_dir += '{}'.format('_predictor')

if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

pkl_name = os.path.join(args.res_dir, args.data_name + '.pkl')

# check whether to load pre-stored pickle data
if os.path.isfile(pkl_name) and not args.reprocess:
    with open(pkl_name, 'rb') as f:
        train_data, test_data, graph_args = pickle.load(f)
# otherwise process the raw data and save t o .pkl
else:
    # determine data formats according to models, NGAE: igraph
    if args.model.startswith('NGAE'):
        input_fmt = 'igraph'
    if args.data_type == 'ENAS':
        train_data, test_data, graph_args = load_ENAS_graphs(args.data_name, n_types=args.nvt,
                                                             fmt=input_fmt)
        train_data = graph_with_complexity(train_data)
        test_data = graph_with_complexity(test_data)
    elif args.data_type == 'NASBench':
        train_data, test_data, graph_args = load_NASBench_graphs(args.data_name, n_types=args.nvt,
                                                           fmt=input_fmt)

    with open(pkl_name, 'wb') as f:
        pickle.dump((train_data, test_data, graph_args), f)

'''Define some train/test functions'''
def train(epoch):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    pred_loss = 0
    complex_loss = 0
    shuffle(train_data)
    pbar = tqdm(train_data)
    g_batch = []
    y_batch = []
    complex_batch = []
    for i, (g, y, complexity) in enumerate(pbar):
        g_batch.append(g)
        y_batch.append(y)
        complex_batch.append(complexity)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            g_batch = model._collate_fn(g_batch)

            mu, logvar = model.encode(g_batch)
            # mu, logvar = torch.rand(len(g_batch), args.nz).to(model.get_device()), torch.rand(len(g_batch), args.nz).to(model.get_device())
            loss, recon, kld = model.loss(mu, logvar, g_batch)
            loss_complexity = 0
            if args.predictor:
                y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
                y_pred = model.predictor(mu)
                loss_y = model.mseloss(y_pred, y_batch)
                loss += loss_y
                
                complex_batch = torch.FloatTensor(complex_batch).unsqueeze(1).to(device)
                complex_pred = model.predictor_complexity(mu)
                loss_complexity = model.mseloss(complex_pred, complex_batch)
                loss += loss_complexity
                pbar.set_description('Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f, y_mse: %0.6f, complexity_mse: %0.6f'\
                        % (epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                        kld.item()/len(g_batch), loss_y/len(g_batch), loss_complexity/len(g_batch)))
            else:
                pbar.set_description('Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f' % (
                                     epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                                     kld.item()/len(g_batch)))
            
            # print('%s', count_parameters(model))
            loss.backward()
            
            train_loss += float(loss)
            recon_loss += float(recon)
            kld_loss += float(kld)
            if args.predictor:
                pred_loss += float(loss_y)
            complex_loss += float(loss_complexity)
            optimizer.step()
            g_batch = []
            y_batch = []
            complex_batch = []

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_data)))

    if args.predictor:
        return train_loss, recon_loss, kld_loss, pred_loss, complex_loss
    return train_loss, recon_loss, kld_loss, complex_loss


def visualize_recon(epoch):
    model.eval()
    # draw some reconstructed train/test graphs to visualize recon quality
    for i, (g, y, complexity) in enumerate(test_data[:10] + train_data[:10]):
        g_recon = model.encode_decode(g)[0]
        name0 = 'graph_epoch{}_id{}_original'.format(epoch, i)
        plot_DAG(g, args.res_dir, name0, data_type=args.data_type)
        name1 = 'graph_epoch{}_id{}_recon'.format(epoch, i)
        plot_DAG(g_recon, args.res_dir, name1, data_type=args.data_type)


def test():
    # test recon accuracy
    model.eval()
    encode_times = 10
    decode_times = 10
    Nll = 0
    pred_loss = 0
    complexity_loss = 0
    n_perfect = 0
    print('Testing begins...')
    
    pbar = tqdm(test_data)
    g_batch = []
    y_batch = []
    complexity_batch = []
    for i, (g, y, complexity) in enumerate(pbar):
        g_batch.append(g)
        y_batch.append(y)
        complexity_batch.append(complexity)
        if len(g_batch) == args.infer_batch_size or i == len(test_data) - 1:
            g = model._collate_fn(g_batch)
            mu, logvar = model.encode(g)
            _, nll, _ = model.loss(mu, logvar, g)
            pbar.set_description('nll: {:.4f}'.format(nll.item()/len(g_batch)))
            Nll += nll.item()
            if args.predictor:
                y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
                y_pred = model.predictor(mu)
                pred = model.mseloss(y_pred, y_batch)
                pred_loss += pred.item()
            complexity_batch = torch.FloatTensor(complexity_batch).unsqueeze(1).to(device)
            complexity_pred = model.predictor_complexity(mu)
            complexity_pred_loss = model.mseloss(complexity_pred, complexity_batch)
            complexity_loss += complexity_pred_loss.item()

            for _ in range(encode_times):
                z = model.reparameterize(mu, logvar)
                for _ in range(decode_times):
                    g_recon = model.decode(z)
                    n_perfect += sum(is_same_DAG(g0, g1) for g0, g1 in zip(g, g_recon))
            g_batch = []
            y_batch = []
            complexity_batch = []
    Nll /= len(test_data)
    pred_loss /= len(test_data)
    pred_rmse = math.sqrt(pred_loss)
    acc = n_perfect / (len(test_data) * encode_times * decode_times)
    if args.predictor:
        print('Test average recon loss: {0}, recon accuracy: {1:.4f}, pred rmse: {2:.4f}'.format(Nll, acc, pred_rmse))
        return Nll, acc, pred_rmse
    else:
        print('Test average recon loss: {0}, recon accuracy: {1:.4f}'.format(Nll, acc))
        return Nll, acc


def prior_validity(scale_to_train_range=False):
    if scale_to_train_range:
        Z_train, Y_train = extract_latent(train_data)
        z_mean, z_std = Z_train.mean(0), Z_train.std(0)
        z_mean, z_std = torch.FloatTensor(z_mean).to(device), torch.FloatTensor(z_std).to(device)
    n_latent_points = 1000
    decode_times = 10
    n_valid = 0
    print('Prior validity experiment begins...')
    G = []
    G_valid = []
    G_train = [g for (g, y, complexity) in train_data]
    pbar = tqdm(range(n_latent_points))
    cnt = 0
    for i in pbar:
        cnt += 1
        if cnt == args.infer_batch_size or i == n_latent_points - 1:
            z = torch.randn(cnt, model.nz).to(model.get_device())
            if scale_to_train_range:
                z = z * z_std + z_mean  # move to train's latent range
            for j in range(decode_times):
                g_batch = model.decode(z)
                G.extend(g_batch)
                if args.data_type == 'ENAS':
                    for g in g_batch:
                        if is_valid_ENAS(g, graph_args.START_TYPE, graph_args.END_TYPE):
                            n_valid += 1
                            G_valid.append(g)
                elif args.data_type == 'BN':
                    for g in g_batch:
                        if is_valid_BN(g, graph_args.START_TYPE, graph_args.END_TYPE):
                            n_valid += 1
                            G_valid.append(g)
            cnt = 0
    r_valid = n_valid / (n_latent_points * decode_times)
    print('Ratio of valid decodings from the prior: {:.4f}'.format(r_valid))

    G_valid_str = [decode_igraph_to_ENAS(g) for g in G_valid]
    r_unique = len(set(G_valid_str)) / len(G_valid_str) if len(G_valid_str)!=0 else 0.0
    print('Ratio of unique decodings from the prior: {:.4f}'.format(r_unique))

    r_novel = 1 - ratio_same_DAG(G_train, G_valid)
    print('Ratio of novel graphs out of training data: {:.4f}'.format(r_novel))
    return r_valid, r_unique, r_novel


def extract_latent(data):
    model.eval()
    Z = []
    Y = []
    g_batch = []
    for i, (g, y, complexity) in enumerate(tqdm(data)):
        # copy igraph
        # otherwise original igraphs will save the H states and consume more GPU memory
        g_ = g.copy()  
        g_batch.append(g_)
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:
            g_batch = model._collate_fn(g_batch)
            mu, _ = model.encode(g_batch)
            mu = mu.cpu().detach().numpy()
            Z.append(mu)
            g_batch = []
        Y.append(y)
    return np.concatenate(Z, 0), np.array(Y)


'''Extract latent representations Z'''
def save_latent_representations(epoch):
    Z_train, Y_train = extract_latent(train_data)
    Z_test, Y_test = extract_latent(test_data)
    latent_pkl_name = os.path.join(args.res_dir, args.data_name +
                                   '_latent_epoch{}.pkl'.format(epoch))
    latent_mat_name = os.path.join(args.res_dir, args.data_name + 
                                   '_latent_epoch{}.mat'.format(epoch))
    with open(latent_pkl_name, 'wb') as f:
        pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
    print('Saved latent representations to ' + latent_pkl_name)
    scipy.io.savemat(latent_mat_name, 
                     mdict={
                         'Z_train': Z_train, 
                         'Z_test': Z_test, 
                         'Y_train': Y_train, 
                         'Y_test': Y_test
                         }
                     )


def interpolation_exp(epoch, num=5):
    print('Interpolation experiments between two random testing graphs')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir) 
    if args.data_type == 'BN':
        eva = Eval_BN(interpolation_res_dir)
    interpolate_number = 10
    model.eval()
    cnt = 0
    for i in range(0, len(test_data), 2):
        cnt += 1
        (g0, _), (g1, _) = test_data[i], test_data[i+1]
        z0, _ = model.encode(g0)
        z1, _ = model.encode(g1)
        print('norm of z0: {}, norm of z1: {}'.format(torch.norm(z0), torch.norm(z1)))
        print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
        Z = []  # to store all the interpolation points
        for j in range(0, interpolate_number + 1):
            zj = z0 + (z1 - z0) / interpolate_number * j
            Z.append(zj)
        Z = torch.cat(Z, 0)
        # decode many times and select the most common one
        G, G_str = decode_from_latent_space(Z, model, return_igraph=True,
                                            data_type=args.data_type) 
        names = []
        scores = []
        for j in range(0, interpolate_number + 1):
            namej = 'graph_interpolate_{}_{}_of_{}'.format(i, j, interpolate_number)
            namej = plot_DAG(G[j], interpolation_res_dir, namej, backbone=True, 
                             data_type=args.data_type)
            names.append(namej)
            if args.data_type == 'BN':
                scorej = eva.eval(G_str[j])
                scores.append(scorej)
        fig = plt.figure(figsize=(120, 20))
        for j, namej in enumerate(names):
            imgj = mpimg.imread(namej)
            fig.add_subplot(1, interpolate_number + 1, j + 1)
            plt.imshow(imgj)
            if args.data_type == 'BN':
                plt.title('{}'.format(scores[j]), fontsize=40)
            plt.axis('off')
        plt.savefig(os.path.join(args.res_dir, 
                    args.data_name + '_{}_interpolate_exp_ensemble_epoch{}_{}.pdf'.format(
                    args.model, epoch, i)), bbox_inches='tight')
        '''
        # draw figures with the same height
        new_name = os.path.join(args.res_dir, args.data_name + 
                                '_{}_interpolate_exp_ensemble_{}.pdf'.format(args.model, i))
        combine_figs_horizontally(names, new_name)
        '''
        if cnt == num:
            break


def interpolation_exp2(epoch):
    if args.data_type != 'ENAS':
        return
    print('Interpolation experiments between flat-net and dense-net')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation2')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir) 
    interpolate_number = 10
    model.eval()
    n = graph_args.max_n
    g0 = [[1]+[0]*(i-1) for i in range(1, n-1)]  # this is flat-net
    g1 = [[1]+[1]*(i-1) for i in range(1, n-1)]  # this is dense-net

    g0, _ = decode_ENAS_to_igraph(str(g0))
    g1, _ = decode_ENAS_to_igraph(str(g1))
    z0, _ = model.encode(g0)
    z1, _ = model.encode(g1)
    print('norm of z0: {}, norm of z1: {}'.format(torch.norm(z0), torch.norm(z1)))
    print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
    Z = []  # to store all the interpolation points
    for j in range(0, interpolate_number + 1):
        zj = z0 + (z1 - z0) / interpolate_number * j
        Z.append(zj)
    Z = torch.cat(Z, 0)
    # decode many times and select the most common one
    G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type)  
    names = []
    for j in range(0, interpolate_number + 1):
        namej = 'graph_interpolate_{}_of_{}'.format(j, interpolate_number)
        namej = plot_DAG(G[j], interpolation_res_dir, namej, backbone=True, 
                         data_type=args.data_type)
        names.append(namej)
    fig = plt.figure(figsize=(120, 20))
    for j, namej in enumerate(names):
        imgj = mpimg.imread(namej)
        fig.add_subplot(1, interpolate_number + 1, j + 1)
        plt.imshow(imgj)
        plt.axis('off')
    plt.savefig(os.path.join(args.res_dir, 
                args.data_name + '_{}_interpolate_exp2_ensemble_epoch{}.pdf'.format(
                args.model, epoch)), bbox_inches='tight')


def interpolation_exp3(epoch):
    if args.data_type != 'ENAS':
        return
    print('Interpolation experiments around a great circle')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation3')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir) 
    interpolate_number = 36
    model.eval()
    n = graph_args.max_n
    g0 = [[1]+[0]*(i-1) for i in range(1, n-1)]  # this is flat-net
    g0, _ = decode_ENAS_to_igraph(str(g0))
    z0, _ = model.encode(g0)
    norm0 = torch.norm(z0)
    z1 = torch.ones_like(z0)
    # there are infinite possible directions that are orthogonal to z0,
    # we just randomly pick one from a finite set
    dim_to_change = random.randint(0, z0.shape[1]-1)  # this to get different great circles
    print(dim_to_change)
    z1[0, dim_to_change] = -(z0[0, :].sum() - z0[0, dim_to_change]) / z0[0, dim_to_change]
    z1 = z1 / torch.norm(z1) * norm0
    print('z0: ', z0, 'z1: ', z1, 'dot product: ', (z0 * z1).sum().item())
    print('norm of z0: {}, norm of z1: {}'.format(norm0, torch.norm(z1)))
    print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
    omega = torch.acos(torch.dot(z0.flatten(), z1.flatten()))
    print('angle between z0 and z1: {}'.format(omega))
    Z = []  # to store all the interpolation points
    for j in range(0, interpolate_number + 1):
        theta = 2*math.pi / interpolate_number * j
        zj = z0 * np.cos(theta) + z1 * np.sin(theta)
        Z.append(zj)
    Z = torch.cat(Z, 0)
    # decode many times and select the most common one
    G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type) 
    names = []
    for j in range(0, interpolate_number + 1):
        namej = 'graph_interpolate_{}_of_{}'.format(j, interpolate_number)
        namej = plot_DAG(G[j], interpolation_res_dir, namej, backbone=True, 
                         data_type=args.data_type)
        names.append(namej)
    # draw figures with the same height
    new_name = os.path.join(args.res_dir, args.data_name + 
                            '_{}_interpolate_exp3_ensemble_epoch{}.pdf'.format(args.model, epoch))
    combine_figs_horizontally(names, new_name)


def smooth_exp1(epoch):
    latent_mat_name = os.path.join(args.res_dir, args.data_name + 
                                   '_latent_epoch{}.mat'.format(epoch))
    save_dir = os.path.join(args.res_dir, 'smoothness')
    data = loadmat(latent_mat_name)
    # X_train, Y_train = extract_latent(train_data)
    X_train = data['Z_train']
    torch.manual_seed(123)
    z0 = torch.tensor(torch.randn(1, model.nz), device='cuda:0', requires_grad=False)
    print(z0)
    model.predictor(z0)
    max_xy = 0.3
    #max_xy = 0.6
    x_range = np.arange(-max_xy, max_xy, 0.005)
    y_range = np.arange(max_xy, -max_xy, -0.005)
    n = len(x_range)
    x_range, y_range = np.meshgrid(x_range, y_range)
    x_range, y_range = x_range.reshape((-1, 1)), y_range.reshape((-1, 1))   
    if True:  # select two principal components to visualize
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, whiten=True)
        pca.fit(X_train)
        d1, d2 = pca.components_[0:1], pca.components_[1:2]
        new_x_range = x_range * d1
        new_y_range = y_range * d2
        grid_inputs = torch.FloatTensor(new_x_range + new_y_range).cuda()
    else:
        grid_inputs = torch.FloatTensor(np.concatenate([x_range, y_range], 1)).cuda()
        if args.nz > 2:
                grid_inputs = torch.cat([grid_inputs, z0[:, 2:].expand(grid_inputs.shape[0], -1)], 1)
    valid_arcs_grid = []
    batch = 3000
    max_n = graph_args.max_n
    data_type = args.data_type
    for i in range(0, grid_inputs.shape[0], batch):
        batch_grid_inputs = grid_inputs[i:i+batch, :]
        # valid_arcs_grid += decode_from_latent_space(batch_grid_inputs, model, 10, max_n, False, data_type) 
        valid_arcs_grid += batch_grid_inputs
    
    print("Evaluating 2D grid points")
    print("Total points: " + str(grid_inputs.shape[0]))
    grid_scores = []
    x, y = [], []
    load_module_state(model, os.path.join(args.res_dir, 
                                            'model_checkpoint{}.pth'.format(300)))
    load_module_state(optimizer, os.path.join(args.res_dir, 
                                            'optimizer_checkpoint{}.pth'.format(300)))
    load_module_state(scheduler, os.path.join(args.res_dir, 
                                            'scheduler_checkpoint{}.pth'.format(300)))
    for i in range(len(valid_arcs_grid)):
        arc = valid_arcs_grid[ i ] 
        if arc is not None:
            score = model.predictor(arc)[0].cpu().detach().numpy()
            x.append(x_range[i, 0])
            y.append(y_range[i, 0])
            grid_scores.append(score)
        else:
            score = 0
            print(i)
    vmin, vmax = np.min(grid_scores), np.max(grid_scores)
    ticks = np.linspace(vmin, vmax, 9, dtype=float).tolist()
    cmap = plt.cm.get_cmap('viridis')
    #f = plt.imshow(grid_y, cmap=cmap, interpolation='nearest')
    sc = plt.scatter(x, y, c=grid_scores, cmap=cmap, vmin=vmin, vmax=vmax, s=10)
    plt.colorbar(sc, ticks=ticks)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_dir + "2D_vis_{}.pdf".format(epoch))

def smoothness_exp(epoch, gap=0.05):
    print('Smoothness experiments around a latent vector')
    smoothness_res_dir = os.path.join(args.res_dir, 'smoothness')
    if not os.path.exists(smoothness_res_dir):
        os.makedirs(smoothness_res_dir) 
    
    #z0 = torch.zeros(1, model.nz).to(device)  # use all-zero vector as center
    
    if args.data_type == 'ENAS': 
        g_str = '4 4 0 3 0 0 5 0 0 1 2 0 0 0 0 5 0 0 0 1 0'  # a 6-layer network
        row = [int(x) for x in g_str.split()]
        row = flat_ENAS_to_nested(row, model.max_n-2)
        g0, _ = decode_ENAS_to_igraph(row)
    elif args.data_type == 'BN':
        g0 = train_data[20][0]
    z0, _ = model.encode(g0)

    # select two orthogonal directions in latent space
    tmp = np.random.randn(z0.shape[1], z0.shape[1])
    Q, R = qr(tmp)
    dir1 = torch.FloatTensor(tmp[0:1, :]).to(device)
    dir2 = torch.FloatTensor(tmp[1:2, :]).to(device)

    # generate architectures along two orthogonal directions
    grid_size = 13
    grid_size = 9
    mid = grid_size // 2
    Z = []
    pbar = tqdm(range(grid_size ** 2))
    for idx in pbar:
        i, j = divmod(idx, grid_size)
        zij = z0 + dir1 * (i - mid) * gap + dir2 * (j - mid) * gap
        Z.append(zij)
    Z = torch.cat(Z, 0)
    if True:
        G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type)
    else:  # decode by 3 batches in case of GPU out of memory 
        Z0, Z1, Z2 = Z[:len(Z)//3, :], Z[len(Z)//3:len(Z)//3*2, :], Z[len(Z)//3*2:, :]
        G = []
        G += decode_from_latent_space(Z0, model, return_igraph=True, data_type=args.data_type)[0]
        G += decode_from_latent_space(Z1, model, return_igraph=True, data_type=args.data_type)[0]
        G += decode_from_latent_space(Z2, model, return_igraph=True, data_type=args.data_type)[0]
    names = []
    for idx in pbar:
        i, j = divmod(idx, grid_size)
        pbar.set_description('Drawing row {}/{}, col {}/{}...'.format(i+1, 
                             grid_size, j+1, grid_size))
        nameij = 'graph_smoothness{}_{}'.format(i, j)
        nameij = plot_DAG(G[idx], smoothness_res_dir, nameij, data_type=args.data_type)
        names.append(nameij)
    #fig = plt.figure(figsize=(200, 200))
    if args.data_type == 'ENAS':
        fig = plt.figure(figsize=(50, 50))
    elif args.data_type == 'BN':
        fig = plt.figure(figsize=(30, 30))
    
    nrow, ncol = grid_size, grid_size
    for ij, nameij in enumerate(names):
        imgij = mpimg.imread(nameij)
        fig.add_subplot(nrow, ncol, ij + 1)
        plt.imshow(imgij)
        plt.axis('off')
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.savefig(os.path.join(args.res_dir, 
                args.data_name + '_{}_smoothness_ensemble_epoch{}_gap={}_small.pdf'.format(
                args.model, epoch, gap)), bbox_inches='tight')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if not args.only_test and not args.only_search:
    # delete old files in the result directory
    remove_list = [f for f in os.listdir(args.res_dir) if not f.endswith(".pkl") and 
            not f.startswith('train_graph') and not f.startswith('test_graph') and
            not f.endswith('.pth')]
    for f in remove_list:
        tmp = os.path.join(args.res_dir, f)
        if not os.path.isdir(tmp) and not args.keep_old:
            os.remove(tmp)

    if not args.keep_old:
        # backup current .py files
        copy('train.py', args.res_dir)
        copy('models.py', args.res_dir)
        copy('util.py', args.res_dir)

    # save command line input
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')

# construct train data
if args.no_test:
    train_data = train_data + test_data

if args.small_train:
    train_data = train_data[:100]

'''Prepare the model'''
# model
model = NGAE(
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

predictor_complexity = nn.Sequential(
        nn.Linear(args.nz, args.hs), 
        nn.Tanh(), 
        nn.Linear(args.hs, 1),
        nn.Sigmoid()
        )
model.predictor_complexity = predictor_complexity
model.mseloss = nn.MSELoss(reduction='sum')

# total_ops, total_params = profile(model, ([train_data[0][0].copy()],), verbose=False)

# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
model.to(device)

if args.all_gpus:
    net = custom_DataParallel(model, device_ids=range(torch.cuda.device_count()))

if args.load_latest_model:
    load_module_state(model, os.path.join(args.res_dir, 'latest_model.pth'))
else:
    if args.continue_from is not None:
        epoch = args.continue_from
        load_module_state(model, os.path.join(args.res_dir, 
                                              'model_checkpoint{}.pth'.format(epoch)))
        load_module_state(optimizer, os.path.join(args.res_dir, 
                                                  'optimizer_checkpoint{}.pth'.format(epoch)))
        load_module_state(scheduler, os.path.join(args.res_dir, 
                                                  'scheduler_checkpoint{}.pth'.format(epoch)))

'''only test to output reconstruct loss'''
if args.only_test:
    epoch = args.continue_from
    #sampled = model.generate_sample(args.sample_number)
    #save_latent_representations(epoch)
    #visualize_recon(300)
    #interpolation_exp2(epoch)
    #interpolation_exp3(epoch)
    # prior_validity(True)
    test()
    # smoothness_exp(epoch, 0.1)
    #smoothness_exp(epoch, 0.05)
    #interpolation_exp(epoch)
    # save_latent_representations(epoch)
    smooth_exp1(epoch)

    # pdb.set_trace()
    sys.exit(0)

'''search network based on latent space'''
if args.only_search:
    print('Begin searching ...')
    if args.search_strategy == 'random':
        graphs = model.random_search(args.search_samples, args.search_optimizer, 1e+1)
        search_dir = os.path.join(args.res_dir, 'random_search')
    else:
        graphs = model.optimal_search(args.search_samples, args.search_optimizer, train_data + test_data, 1e+3)
        search_dir = os.path.join(args.res_dir, 'optimal_search')
    
    if not os.path.exists(search_dir):
        os.makedirs(search_dir)

    print(search_dir)
    for i, g in enumerate(graphs):
        namei = '{}_graph_{}'.format(args.search_optimizer, i)
        plot_DAG(g[0], search_dir, namei, data_type=args.data_type)
    
    finded_graphs_path = os.path.join(search_dir, 'finded_graphs.pkl')
    with open(finded_graphs_path, 'wb') as f:
        pickle.dump(graphs, f)
    sys.exit(0)

'''train from scratch'''
if args.train_from_scratch:
    if args.search_strategy == 'random':
        search_dir = os.path.join(args.res_dir, 'random_search')
    else:
        search_dir = os.path.join(args.res_dir, 'optimal_search')
    finded_graphs_path = os.path.join(search_dir, 'finded_graphs.pkl')
    if os.path.getsize(finded_graphs_path) > 0:
        with open(finded_graphs_path, 'rb') as f:
            graphs = pickle.load(f)

    valid_graphs = []
    if args.data_type == 'ENAS':
        for g in graphs:
            if is_valid_ENAS(g[0], model.START_TYPE, model.END_TYPE):
                decoded_vector = decode_igraph_to_ENAS(g[0])
                valid_graphs.append((decoded_vector, g[1]))
                namei = 'train_model_{}_graph_{}'.format(args.search_optimizer, decoded_vector)
                plot_DAG(g[0], search_dir, namei, data_type=args.data_type)
    
    enas_pos = './software/enas/'
    scores = []
    for g in valid_graphs:
        arc = g[0]
        print('Fully training ENAS architecture ' + arc)
        save_appendix = ''.join(arc.split())
        if not os.path.exists('/home/lijian/data/outputs_' + save_appendix):
            pwd = os.getcwd()
            os.chdir(enas_pos)
            os.system('CUDA_VISABLE_DEVICES={} ./scripts/custom_cifar10_macro_final_6.sh'.format(0) + ' "' + arc + '" ' + save_appendix)
            os.chdir(pwd)
        with open('/home/lijian/data/outputs_' + save_appendix + '/stdout', 'r') as f:
            last_line = f.readlines()[-1]
            scores.append(last_line)

    print()
    print('Fully trained architecture, WS acc, and test acc:')
    for i, g in enumerate(valid_graphs):
        print(g[0], g[1].data.cpu().numpy(), scores[i])
    print('Average score is {}'.format(np.mean([float(x.split()[1]) for x in scores])))
    sys.exit(0)

# plot sample train/test graphs
if not os.path.exists(os.path.join(args.res_dir, 'train_graph_id0.pdf')) or args.reprocess:
    if not args.keep_old:
        for data in ['train_data', 'test_data']:
            G = [g for g, y in eval(data)[:10]]
            for i, g in enumerate(G):
                name = '{}_graph_id{}'.format(data[:-5], i)
                plot_DAG(g, args.res_dir, name, data_type=args.data_type)

'''Training begins here'''
min_loss = math.inf  # >= python 3.5
min_loss_epoch = None
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
test_results_name = os.path.join(args.res_dir, 'test_results.txt')
if os.path.exists(loss_name) and not args.keep_old:
    os.remove(loss_name)

start_epoch = args.continue_from if args.continue_from is not None else 0
for epoch in range(start_epoch + 1, args.epochs + 1):
    if args.predictor:
        train_loss, recon_loss, kld_loss, pred_loss, complex_loss = train(epoch)
    else:
        train_loss, recon_loss, kld_loss, complex_loss = train(epoch)
        pred_loss = 0.0
    with open(loss_name, 'a') as loss_file:
        loss_file.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
            train_loss/len(train_data), 
            recon_loss/len(train_data), 
            kld_loss/len(train_data), 
            pred_loss/len(train_data), 
            complex_loss/len(train_data)
            ))

    scheduler.step(train_loss)
    if epoch % args.save_interval == 0:
        print("save current model...")
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
        scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_name)
        torch.save(optimizer.state_dict(), optimizer_name)
        torch.save(scheduler.state_dict(), scheduler_name)
        print("visualize reconstruction examples...")
        visualize_recon(epoch)
        print("extract latent representations...")
        save_latent_representations(epoch)
        print("sample from prior...")
        sampled = model.generate_sample(args.sample_number)
        for i, g in enumerate(sampled):
            namei = 'graph_{}_sample{}'.format(epoch, i)
            plot_DAG(g, args.res_dir, namei, data_type=args.data_type)
        print("plot train loss...")
        losses = np.loadtxt(loss_name)
        if losses.ndim == 1:
            continue
        fig = plt.figure()
        num_points = losses.shape[0]
        plt.plot(range(1, num_points+1), losses[:, 0], label='Total')
        plt.plot(range(1, num_points+1), losses[:, 1], label='Recon')
        plt.plot(range(1, num_points+1), losses[:, 2], label='KLD')
        plt.plot(range(1, num_points+1), losses[:, 3], label='Pred')
        plt.xlabel('Epoch')
        plt.ylabel('Train loss')
        plt.legend()
        plt.savefig(loss_plot_name)

'''Testing begins here'''
if args.predictor:
    Nll, acc, pred_rmse = test()
else:
    Nll, acc = test()
    pred_rmse = 0
r_valid, r_unique, r_novel = prior_validity(True)
with open(test_results_name, 'a') as result_file:
    result_file.write("Epoch {} Test recon loss: {} recon acc: {:.4f} r_valid: {:.4f}".format(
            epoch, Nll, acc, r_valid) + 
            " r_unique: {:.4f} r_novel: {:.4f} pred_rmse: {:.4f}\n".format(
            r_unique, r_novel, pred_rmse))
interpolation_exp2(epoch)
smoothness_exp(epoch)
interpolation_exp3(epoch)

pdb.set_trace()