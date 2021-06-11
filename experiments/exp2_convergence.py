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
                pred = line.split('loss: ')[1].split(', recon: ')[1].split(', kld: ')[1].split(', pred: ')[1].split(': ')[0]
                loss_file.write(loss + ' ' + recon + ' ' + kld + ' ' + pred + '\n')

losses = np.loadtxt(loss_name)
losses = np.sqrt(np.array([loss for i, loss in enumerate(losses) if i%1071==0 and i > 0]))
num_points = losses.shape[0]

fig = plt.figure()
plt.plot(range(1, num_points+1), losses[:, 3], label='Train')
plt.scatter([100, 200, 300], [0.0037085547451984058, 0.00261684295357763, 0.0021900033144668586], label='Test', c='red', marker='x')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.xticks([0, 100, 200, 300], [0, 100, 200, 300])
plt.legend()
plt.tight_layout()
plt.savefig('experiments/exp2_pred.pdf')