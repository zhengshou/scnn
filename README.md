# Segment-CNN

By Zheng Shou, Dongang Wang, and Shih-Fu Chang.

### Introduction

Segment-CNN (S-CNN) is a segment-based deep learning framework for temporal action localization in untrimmed long videos.

This code has been tested on Ubuntu 14.04 with NVIDIA GTX 980.

### License

S-CNN is released under the MIT License (refer to the LICENSE file for details).

### Citing

If you find S-CNN useful, please consider citing:

    @inproceedings{scnn_shou_wang_chang_cvpr16,
      author = {Zheng Shou and Dongang Wang and Shih-Fu Chang},
      title = {Temporal Action Localization in Untrimmed Videos via Multi-stage CNNs},
      year = {2016},
      booktitle = {CVPR} 
      }
    
We build this repo based on C3D and THUMOS Challenge 2014 . Please cite the following papers as well:

D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, Learning Spatiotemporal Features with 3D Convolutional Networks, ICCV 2015.

Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell, Caffe: Convolutional Architecture for Fast Feature Embedding, arXiv 2014.

A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar, and L. Fei-Fei, Large-scale Video Classification with Convolutional Neural Networks, CVPR 2014.

    @misc{THUMOS14,
      author = "Jiang, Y.-G. and Liu, J. and Roshan Zamir, A. and Toderici, G. and Laptev, I. and Shah, M. and Sukthankar, R.",
      title = "{THUMOS} Challenge: Action Recognition with a Large Number of Classes",
      howpublished = "\url{http://crcv.ucf.edu/THUMOS14/}",
      Year = {2014}
      }

### Installation:
0. Download ffmpeg from https://www.ffmpeg.org/ to `./lib/preprocess/`
1. Compile 3D CNN:
    - Compile C3D_sample_rate, which is used for the proposal network and classification network
    - Compile C3D_overlap_loss, which is used for the localization network
    - Hint: please refer to [C3D](https://github.com/facebook/C3D) and [Caffe](https://github.com/BVLC/caffe) for more details about compilation
2. Download pre-trained models to `./models/`: either from [Dropbox](https://www.dropbox.com/s/657cuo60xg41zln/models.7z?dl=0) or [Baiduyun](http://pan.baidu.com/s/1o8AHrUa)

### Run demo:
0. change to demo directory: `cd ./demo/`.
1. run the demo using the matlab code `run_demo.m` or the python code `run_demo.py`.
2. find the final result in the folder `./pred/final/`. either in .mat format (for matlab) or .csv format (for python).

### Our pre-trained models and pre-computed results of S-CNN on THUMOS Challenge 2014 action detection task:
0. Models:
    - `./models/conv3d_deepnetA_sport1m_iter_1900000`: C3D model pre-trained on Sports1M dataset by Tran et al;
    - `./models/THUMOS14/proposal/snapshot/SCNN_uniform16_binary_iter_30000`: our trained S-CNN proposal network; 
    - `./models/THUMOS14/classification/snapshot/SCNN_uniform16_cls20_iter_30000`: our trained S-CNN classification network; 
    - `./models/THUMOS14/localization/snapshot/SCNN_uniform16_cls20_with_overlap_loss_iter_30000`: our trained S-CNN localization network.
1. Results:
    - `./experiments/THUMOS14/network_proposal/result/res_seg_swin.mat`: contains the output results of the proposal network; 
    - Note for the meaning of each column in the following res_seg_swin.mat: 
        * 1: video name in THUMOS14 test set
        * 3: start frame index
        * 4: end frame index
        * 5: start time
        * 6: end time
        * 10: confidence score of being action
        * 12: sliding window overlap. all 0.25. means 75% overlap window.
    - `./experiments/THUMOS14/network_localization/result/res_seg_swin.mat`: contains the output results of the localization network;
    - evaluate mAP: run `./experiments/THUMOS14/eval/eval_scnn_thumos14.m` and results are stored in `./experiments/THUMOS14/eval/res_scnn_thumos14.mat`.

### Train your own model
0. Please refer to [C3D](https://github.com/facebook/C3D) and [Caffe](https://github.com/BVLC/caffe) for more general instructions about how to train 3D CNN model.
1. 






