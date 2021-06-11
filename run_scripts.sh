time=$(date "+%Y%m%d")
save_appendix = $("_NGAE_sigmoid_${time}") # "_NGAE"
echo ${save_appendix}

## only train VAE
nohup python train.py --data-name final_structures6 --save-interval 100 --save-appendix ${save_appendix} --epochs 300 --lr 1e-4 --model NGAE --bidirectional --nz 56 --batch-size 32 >> "./logs/train_VAE_${time}.log" 2>&1 &

## train VAE + predictor
nohup python train.py --data-name final_structures6 --save-interval 100 --save-appendix ${save_appendix} --epochs 300 --lr 1e-4 --model NGAE --predictor --bidirectional --nz 56 --batch-size 32 >> "./logs/train_VAE_predictor_${time}.log" 2>&1 &

## train VAE + predictor + complexity (NGAE)
nohup python train.py --data-name final_structures6 --save-interval 100 --save-appendix ${save_appendix} --epochs 300 --lr 1e-4 --model NGAE --predictor --bidirectional --nz 56 --batch-size 32 >> "./logs/train_NGAE_${time}.log" 2>&1 &

## test rec acc and rmse
nohup python train.py --data-name final_structures6 --save-interval 100 --save-appendix ${save_appendix} --epochs 300 --lr 1e-4 --model NGAE --predictor --bidirectional --nz 56 --batch-size 32 --only-test --continue-from 300 >> "./logs/test_${time}.log" 2>&1 &

## optimal search
python train.py --data-name final_structures6 --save-interval 100 --save-appendix ${save_appendix} --epochs 300 --lr 1e-4 --model NGAE --bidirectional --nz 56 --batch-size 32 --only-search --search-optimizer sgd --search-strategy optimal --search-samples 10 --continue-from 300

## random search
python train.py --data-name final_structures6 --save-interval 100 --save-appendix ${save_appendix} --epochs 300 --lr 1e-4 --model NGAE --predictor --bidirectional --nz 56 --batch-size 32 --only-search --search-optimizer sgd --search-strategy random --search-samples 10 --continue-from 300

## train from scratch for random
python train.py --data-name final_structures6 --save-interval 100 --save-appendix ${save_appendix} --epochs 300 --lr 1e-4 --model NGAE --predictor --bidirectional --nz 56 --batch-size 32 --search-optimizer sgd --search-strategy random --search-samples 10 --continue-from 300 --train-from-scratch

## train from scratch for optimal
python train.py --data-name final_structures6 --save-interval 100 --save-appendix ${save_appendix}_sigmoid --epochs 300 --lr 1e-4 --model NGAE --predictor --bidirectional --nz 56 --batch-size 32 --train-from-scratch --search-strategy optimal