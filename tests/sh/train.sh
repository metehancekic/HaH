#!/bin/bash

export CUDA_VISIBLE_DEVICES="2"


COMMAND="python -m tests.train_model --multirun
                train.type=standard
                nn.classifier=HaH_VGG
                nn.conv_layer_type=noisy
                nn.threshold=-0.2
                nn.divisive.sigma=0.1
                nn.sigma_noise=0.2,0.1
                nn.lr=0.001
                train.epochs=100
                train.regularizer.l1_weight.scale=0.001
                train.regularizer.active=['hah']
                train.regularizer.hah.layer=Conv2d
                train.regularizer.hah.alpha=[0.0045,0.0025,0.0013,0.001,0.0008,0.0005,0.0,0.0,0.0,0.0,0.0,0.0,0.0]"
echo $COMMAND
eval $COMMAND

# COMMAND="python -m src.train_model --multirun
#                 train.type=standard
#                 nn.classifier=HaH_VGG
#                 nn.conv_layer_type=implicitconv2d
#                 nn.threshold=0.0
#                 nn.divisive.sigma=none
#                 nn.hah_layer_num=0
#                 nn.lr=0.001
#                 train.epochs=100
#                 train.regularizer.l1_weight.scale=0.001
#                 train.regularizer.active=none
#                 train.regularizer.hah.layer=Conv2d
#                 train.regularizer.hah.alpha=[0.0045,0.0025,0.0013,0.001,0.0008,0.0005,0.0,0.0,0.0,0.0,0.0,0.0,0.0]"
# echo $COMMAND
# eval $COMMAND
