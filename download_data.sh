#!/bin/bash
cd data
wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip
unzip vgg.zip
rm vgg.zip
wget http://kitti.is.tue.mpg.de/kitti/data_road.zip
unzip data_road.zip
rm data_road.zip
pip install tqdm