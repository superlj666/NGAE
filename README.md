Efficient Neural Architecture Search via Graph VAE
===============================================================================

About
-----

Differentiable neural architecture search (NAS) methods manage to map the discrete and non-differentiable search space to continuous space and train the search model on this continuous space via gradient optimization. The continuous representation of the discrete neural networks is the key to discovering novel architectures, but existing differentiable NAS methods fail to characterize the topological information of the candidate networks. In this paper, we propose an efficient differentiable NAS approach, of which the latent space is built upon variational autoencoder (VAE) and graph neural networks (GNN). In the training stage, the framework is trained jointly to optimize four components in an end-to-end manner: the encoder, the performance predictor, the complexity predictor, and the decoder. The encoder and the decoder can be regarded as a graph VAE to construct the latent continuous search space. Note that, the proposed framework can find neural networks with good performance and low complexity owing to the use of two predictors that fit the performance and computational complexity on the continuous space, respectively. Extensive experiments demonstrate our framework not only produces smooth continuous representations but also discovers powerful neural architectures with both higher accuracy and fewer parameters.

The implementation of ***NGAE*** is mainly based on [D-VAE](https://github.com/muhanzhang/D-VAE).

Installation
------------

Tested with Python 3.7, PyTorch 1.4.0 and CUDA 10.0.130.

Install [PyTorch](https://pytorch.org/) >= 1.4.0

Install python-igraph by:

    pip install python-igraph

Install pygraphviz by:

    conda install graphviz
    conda install pygraphviz

Other required python libraries: tqdm, six, scipy, numpy, matplotlib


Usage
--------

### Train VAE + $f_{perf}$ + $f_{cmplx}$ (NGAE)
    python train.py --data-name final_structures6 --save-interval 100 --save-appendix _NGAE --epochs 300 --lr 1e-4 --model NGAE --predictor --bidirectional --nz 56 --batch-size 32

### Test reconstruction error and RMSE
    python train.py --data-name final_structures6 --save-interval 100 --save-appendix _NGAE --epochs 300 --lr 1e-4 --model NGAE --predictor --bidirectional --nz 56 --batch-size 32 --only-test --continue-from 300

### Inference
    python train.py --data-name final_structures6 --save-interval 100 --save-appendix _NGAE --epochs 300 --lr 1e-4 --model NGAE --bidirectional --nz 56 --batch-size 32 --only-search --search-optimizer sgd --search-strategy optimal --search-samples 10 --continue-from 300 

### Train from scratch
    python train.py --data-name final_structures6 --save-interval 100 --save-appendix _NGAE_sigmoid --epochs 300 --lr 1e-4 --model NGAE --predictor --bidirectional --nz 56 --batch-size 32 --train-from-scratch --search-strategy optimal

More scripts can be founded in `run_scripts.sh`

Dependency
---------------------
- python 3.7.9

- pytorch = 1.7.1

- torchvision = 0.8.2

- numpy = 1.19.2

- matplotlib = 3.3.2

All experiments are conducted on a Linux server equipped with one NVIDIA Tesla P100 with 16 GB memory.