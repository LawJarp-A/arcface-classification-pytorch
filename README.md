# arcface-classification-pytorch
Implementation of a classification model that incorporates arcface loss and GeM pooling to improve intra class closeness and inter class seperability. PyTorch is primarily used along with fastai framework to speed up development and benchmarking.

## Arcface loss
Refer to [this paper](https://paperswithcode.com/method/arcface) for more information 

## How to set the parameters for Arcface loss
* There are a few hyper parameters that are dependent on the number of classes and how much error one can sustain. 
* Refer [this wonderful article](https://hav4ik.github.io/articles/deep-metric-learning-survey#adacos) about using arcface and it's components and how to choose the right values for the hyper parameters
