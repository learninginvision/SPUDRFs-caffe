# Self-Paced Deep Regression Forests with Consideration on Underrepresented Samples

This repo is for the 2020 ECCV paper [**Self-Paced Deep Regression Forests with Consideration on Underrepresented Samples**](https://arxiv.org/abs/2004.01459v4). In this paper we propose a new deep discriminative model – self-paced deep regression forests considering sample uncertainty (SPUDRFs) based on self-pace learning (SPL). It builds up a new self-paced learning paradigm: easy and underrepresented samples first. This paradigm could be extended to combine with a variety of deep discriminative models. Extensive experiments on two computer vision tasks, i.e., facial age estimation and head pose estimation, demonstrate the efficacy of SPUDRFs, where state-of-the-art performances are achieved.      

![figure1](https://github.com/learninginvision/SPUDRFs/blob/master/pic/Figure1.png)   

## Performance: ##  
- MORPH II   
The MAEs of SP-DRFs and SPUDRFs at each pace descends gradually. The SPUDRFs show its superiority of taking predictive uncertainty into consideration, when compared with SP-DRFs.   


- Biwi  
The leaf node distribution of SP-DRFs and SPUDRFs in gradual learning process. Three paces, i.e. pace 1, 3, and 6, are randomly chosen for visualization. For
SP-DRFs, the Gaussian means of leaf nodes (the red points in the second row) are concentrated in a small range, incurring seriously biased solutions. For SPUDRFs, the Gaussian means of leaf nodes (the orange points in the third row) distribute widely, leading to much better MAE performance.   

## Getting Started：  

* **Clone this repo:**  
> git clone https://github.com/learninginvision/SPUDRFs   
cd SPUDFRs   

* **Train SPUDRFs:**  
You can train your SPUDRFs from scratch easily by running **main.py**. Here is a description of the main codes.  
    - **main.py:**   
train SPUDRFs from scratch  
    - **train.py:**   
complete one pace training for given train set
    - **predict.py:**   
complete a test for given test set
    - **picksamples.py:**   
get the train set for next pace  

## Transplant:

Like [DRFs](https://github.com/shenwei1231/caffe-DeepRegressionForests), if you have a different Caffe or CUDA version than this repository and would like to try out the proposed SPUDRFs layers, you can transplant the following code to your repository.

(util) 

- include/caffe/util/sampling.hpp
- src/caffe/util/sampling.cpp
- include/caffe/util/neural_decision_util_functions.hpp
- src/caffe/util/neural_decision_util_functions.cu

(training) 

- include/caffe/layers/neural_decision_reg_forest_loss_layer.hpp 
- src/caffe/layers/neural_decision_reg_forest_loss_layer.cpp
- src/caffe/layers/neural_decision_reg_forest_loss_layer.cu

- include/caffe/layers/neural_decision_reg_forest_layer.hpp 
- src/caffe/layers/neural_decision_reg_forest_layer.cpp
- src/caffe/layers/neural_decision_reg_forest_layer.cu
