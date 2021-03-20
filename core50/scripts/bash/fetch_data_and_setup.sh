#!/usr/bin/env bash

# Setup
DIR="$( cd "$( dirname "$0" )" && pwd )"
mkdir $DIR/../../data
mkdir $DIR/../../data/logs
mkdir $DIR/../../data/snapshots

echo "Downloading caffenet pre-trained model..."
wget --directory-prefix=$DIR'/../../data/' http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

echo "Downloading Core50 (128x128 version)..."
wget --directory-prefix=$DIR'/../../data/' http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip

echo "Unzipping Core50..."
unzip $DIR/../../data/core50_128x128.zip -d $DIR/../../data/

# One can get the pre-trained model for ResNet-10 at
# https://drive.google.com/file/d/0B6VgjAr4t_oTdUhaclljRWlPSVU/view?usp=sharing
# Put the model at the path "core50/data/".

# One can get the pre-trained model for ResNet-50 at
# https://drive.google.com/open?id=0B6VgjAr4t_oTTDh2SVJIa2VkZVU
# Put the model at the path "core50/data/".
