import numpy as np
import matplotlib.pyplot as plt
import math
import os


with open('./logs/0524_sigmoid.log', 'r') as f:
    lines = f.readlines()

loss_name = './experiments/train_loss.txt'
if not os.path.exists(loss_name):
    with open(loss_name, 'a') as loss_file:
        for line in lines:
            if line.startswith('Epoch: '):
                loss = line.split('loss: ')[1].split(', recon: ')[0]
                recon = line.split('loss: ')[1].split(', recon: ')[1].split(', kld: ')[0]
                kld = line.split('loss: ')[1].split(', recon: ')[1].split(', kld: ')[1].split(', pred: ')[0]
                pred = line.split('loss: ')[1].split(', recon: ')[1].split(', kld: ')[1].split(', pred: ')[0]
                loss_file.write(loss + ' ' + recon + ' ' + kld + ' ' + pred + '\n')

losses = np.loadtxt(loss_name)
fig = plt.figure()
num_points = losses.shape[0]
# plt.plot(range(1, num_points+1), losses[:, 0], label='Total')
plt.plot(range(1, num_points+1), losses[:, 1], label='Reconstruction Loss')
plt.plot(range(1, num_points+1), losses[:, 2], label='KL Divergence')
# plt.plot(range(1, num_points+1), losses[:, 3], label='Pred')
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.xticks([x for x in np.arange(len(losses)) if x%107100==0], ('0','100', '200', '300'))
# plt.xticks(np.arange(4)*100000, ('0','100k', '200k', '300k'))
plt.legend()
plt.savefig('experiments/exp1_convergence.pdf')