# SiamCAR-V2


# Train

```
sh ./bin/cmd_dist_train.sh 
```

# Test
```
sh ./bin/cmd_test.sh 
```




# Results

```
Backbone Dataset  Accuracy Robustness Lost number
Alexnet  Real     0.801    0.194      15
         Virtual  0.861    0.013      1
         Mix      0.881    0          0
ConvNeXt Real     0.805    0.026      2
         Virtual  0.879    0          0
         Mix      0.891    0          0
```
# Reference
```
https://github.com/ohhhyeahhh/SiamCAR

https://github.com/MegviiDetection/video_analyst

[1] Guo D ,  Wang J ,  Cui Y , et al. SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking[C]//CVPR,2020.

[2] Xu Y, Wang Z, Li Z, et al. SiamFC++: Towards Robust and Accurate Visual Tracking with Target Estimation Guidelines[C]//AAAI,2020.

```
