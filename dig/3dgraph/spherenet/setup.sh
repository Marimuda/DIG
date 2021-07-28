#!/bin/bash
# same with schnet and dimenetpp

conda create -n spherenet python=3.7
conda activate spherenet

conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-geometric
