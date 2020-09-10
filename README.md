# Self-Paced Deep Regression Forests with Consideration on Underrepresented Examples

<div align=center>
<img src="https://github.com/learninginvision/SPUDRFs/blob/master/pic/Figure1.png" width="600">
</div>    


*Abstract*: Deep discriminative models (e.g. deep regression forests, deep neural decision forests) have achieved remarkable success recently to solve problems such as facial age estimation and head pose estimation. Most existing methods pursue robust and unbiased solutions either through learning discriminative features, or reweighting samples. We argue what is more desirable is learning gradually to discriminate like our human beings, and hence we resort to self-paced learning (SPL). Then, a natural question arises: can self-paced regime lead deep discriminative models to achieve more robust and less biased solutions? To this end, we propose a new deep discriminative model—self-paced deep regression forests with consideration on underrepresented examples (SPUDRFs). It tackles the fundamental ranking and selecting problem in SPL from a new perspective: fairness. This paradigm is fundamental and could be easily combined with a variety of deep discriminative models (DDMs). Extensive experiments on two computer vision tasks, i.e., facial age estimation and head pose estimation, demonstrate the efficacy of SPUDRFs, where state-of-the-art performances are achieved.


## Poster Presentation 
Poster Presentation [[PDF]](https://drive.google.com/file/d/1bD8ZTLP_6LxMHBdbNBzn6gEdC2JqUCiL/view?usp=sharing)


## Tasks and Performances  
### **Age Estimation on MORPH II Dataset**   
<div align=center>
<img src="https://github.com/learninginvision/SPUDRFs/blob/master/pic/SPUDRFs_validation_new.png" width="800">
</div>

The gradual learning process of SP-DRFs and SPUDRFs. **Left:** The typical worst cases at each iteration. The two numbers below each image are the real age (left) and predicted age (right). **Right:** The MAEs of SP-DRFs and SPUDRFs at each pace descend gradually. The SPUDRFs show its superiority of taking predictive uncertainty into consideration, when compared with SP-DRFs.

### **Head Pose Estimation on BIWI Dataset**   
<div align=center>
<img src="https://github.com/learninginvision/SPUDRFs/blob/master/pic/Uncertainty_efficacy.png" width="600"> 
</div>

The leaf node distribution of SP-DRFs and SPUDRFs in gradual learning process. Three paces, i.e. pace 1, 3, and 6, are randomly chosen for visualization. For SP-DRFs, the Gaussian means of leaf nodes (the red points in the second row) are concentrated in a small range, incurring seriously biased solutions. For SPUDRFs, the Gaussian means of leaf nodes (the orange points in the third row) distribute widely, leading to much better MAE performance.

## Code 
We provide our implementation of self-paced deep regression forests with consideration on underrepresentd examples.

If you use this code for your research, please cite:

Self-Paced Deep Regression Forests with Consideration on Underrepresented Examples.<br>
Lili Pan, Shijie Ai, Yazhou Ren, Zenglin Xu. In ECCV2020.<br>
[[Bibtex]](https://github.com/learninginvision/SPUDRFs/blob/master/pic/bib.txt)  [[PDF]](https://arxiv.org/abs/2004.01459v4)

## Setup

### requirements:
- caffe
- python 2.7  

### **Clone this repo:**  
```shell
git clone https://github.com/learninginvision/SPUDRFs   
cd SPUDFRs  
```

### Code descritption: 
Here is the description of the main codes.  
- **main.py:**   
train SPUDRFs from scratch  
- **train.py:**   
complete one pace training for a given train set
- **predict.py:**   
complete a test for a given test set
- **picksamples.py:**   
select samples for next pace    

### Train your SPUDRFs from scratch
```shell
cd SPUDRFs/caffe_soft
sudo make clean
sudo make all
sudo make pycaffe
cd ..
python2 main.py
```
You can download the final model from here

## Acknowledgments
This code is inspired by [caffe-DRFs](https://github.com/shenwei1231/caffe-DeepRegressionForests).
