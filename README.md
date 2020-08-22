# Self-Paced Deep Regression Forests with Consideration on Underrepresented Examples

<div align=center>
<img src="https://github.com/learninginvision/SPUDRFs/blob/master/pic/Figure1.png" width="600">
</div>    


Abstract: Deep discriminative models (e.g. deep regression forests, deep neural decision forests) have achieved remarkable success recently to solve problems such as facial age estimation and head pose estimation. Most existing methods pursue robust and unbiased solutions either through learning discriminative features, or reweighting samples. We argue what is more desirable is learning gradually to discriminate like our human beings, and hence we resort to self-paced learning (SPL). Then, a natural question arises: can self-paced regime lead deep discriminative models to achieve more robust and less biased solutions? To this end, we propose a new deep discriminative model—self-paced deep regression forests with consideration on underrepresented examples (SPUDRFs). It tackles the fundamental ranking and selecting problem in SPL from a new perspective: fairness. This paradigm is fundamental and could be easily combined with a variety of deep discriminative models (DDMs). Extensive experiments on two computer vision tasks, i.e., facial age estimation and head pose estimation, demonstrate the efficacy of SPUDRFs, where state-of-the-art performances are achieved.


## Poster Presentation 
Poster Presentation [[PDF]](https://drive.google.com/file/d/1bD8ZTLP_6LxMHBdbNBzn6gEdC2JqUCiL/view?usp=sharing)


## Performance  
### **Age Estimation on MORPH II Dataset**   
The MAEs of SP-DRFs and SPUDRFs at each pace descends gradually. The SPUDRFs show its superiority of taking predictive uncertainty into consideration, when compared with SP-DRFs.   

<div align=center>
<img src="https://github.com/learninginvision/SPUDRFs/blob/master/pic/SPUDRFs_validation_new.png" width="800">
</div>

### **Head Pose Estimation on Biwi Dataset**  
The leaf node distribution of SP-DRFs and SPUDRFs in gradual learning process. Three paces, i.e. pace 1, 3, and 6, are randomly chosen for visualization. For
SP-DRFs, the Gaussian means of leaf nodes (the red points in the second row) are concentrated in a small range, incurring seriously biased solutions. For SPUDRFs, the Gaussian means of leaf nodes (the orange points in the third row) distribute widely, leading to much better MAE performance.   

<div align=center>
<img src="https://github.com/learninginvision/SPUDRFs/blob/master/pic/Uncertainty_efficacy.png" width="600"> 
</div>


## Code 
We provide our implementation of self-paced deep regression forests with consideration on underrepresentd samples.

If you use this code for your research, please cite:

Self-Paced Deep Regression Forests with Consideration on Underrepresented Samples.<br>
Lili Pan, Shijie Ai, Yazhou Ren, Zenglin Xu. In ECCV2020.<br>
[[Bibtex]](https://github.com/learninginvision/SPUDRFs/blob/master/pic/bib.txt)  [[PDF]](https://arxiv.org/abs/2004.01459v4)

## Setup

### **Clone this repo:**  
> git clone https://github.com/learninginvision/SPUDRFs   
cd SPUDFRs   

### **Train SPUDRFs:**  
You can train your SPUDRFs from scratch easily by running **main.py**. Here is a description of the main codes.  
- **main.py:**   
train SPUDRFs from scratch  
- **train.py:**   
complete one pace training for given train set
- **predict.py:**   
complete a test for given test set
- **picksamples.py:**   
select samples for next pace  

## Transplant

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

## Acknowledgments
This code is inspired by caffe-DRFs.
