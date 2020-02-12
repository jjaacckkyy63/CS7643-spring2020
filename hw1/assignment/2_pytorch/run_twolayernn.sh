#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model twolayernn \
    --hidden-dim 512 \
    --epochs 40 \
    --weight-decay 1e-5\
    --momentum 0.9 \
    --batch-size 512 \
    --lr 0.001 | tee twolayernn.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
