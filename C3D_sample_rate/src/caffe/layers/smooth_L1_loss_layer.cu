// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// Updated by Zheng Shou
// ------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
Dtype SmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

    int count = bottom[0]->count();

  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  //LOG(INFO) << "diff_.mutable_cpu_data()[0]: "<<diff_.cpu_data()[0];
  //LOG(INFO) << "diff_.mutable_cpu_data()[1]: "<<diff_.cpu_data()[1];
  
  // has_weights_ is an option to be set in prototxt but we do not implement this
  // Alternatively, we set has_weights_ in smooth_L1_loss_layer.cpp to 1 if (bottom.size() == 3)
  // bottom[2]->cpu_data() are weights in hdf5 input for each reg class: # 2(N-1): 1 1 0 0 0 0 ... (N=4)
  // the reason to be "2"N-1 is there are 2 reg targets for each class. So this two should always be same as 1 or 0
  // We handle setting loss_weight_ and shuffle data when generating all segments/sliding windows
  // Here N doesn't include background class - class 0. so for binary cls, this is 0 0 (no reg) and 1 1 (do reg)
  if (has_weights_) {
    caffe_mul(
        count,
        bottom[2]->cpu_data(),
        diff_.cpu_data(),
        diff_.mutable_cpu_data());  // d := w * (b0 - b1)
  }	  
  
  // f(x) = 0.5 * x^2    if |x| < 1
  //        |x| - 0.5    otherwise
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    //LOG(INFO) << "input_data["<< i <<"]: "<< input_data[i];
    //LOG(INFO) << "target["<< i <<"]: "<< target[i];  
    if (abs(diff_.cpu_data()[i]) < 1) {
      loss += 0.5 * diff_.cpu_data()[i] * diff_.cpu_data()[i];
    } else {
      loss += abs(diff_.cpu_data()[i]) - 0.5;
    }
  }
  
  // loss_weight_ for smooth l1 loss layer
  loss = loss * loss_weight_;

  (*top)[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
  
  LOG(INFO) << "smooth loss: " << loss / bottom[0]->num();
  return loss / bottom[0]->num();
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	
  int count = diff_.count();
  
   for (int i = 0; i < count; ++i) {
  // f'(x) = x         if |x| < 1
  //       = sign(x)   otherwise
    if (abs(diff_.cpu_data()[i]) < 1) {
      diff_.mutable_cpu_data()[i] = diff_.cpu_data()[i];
    } else {
      diff_.mutable_cpu_data()[i] = (Dtype(0) < diff_.cpu_data()[i]) - (diff_.cpu_data()[i] < Dtype(0));
    }
  }
  
  for (int i = 0; i < 2; ++i) {
    if (i == 0) {
      const Dtype sign = (i == 0) ? 1 : -1;
	  // old version doesn't have "top[0]->cpu_diff()[0]"
      // loss_weight_ for smooth l1 loss layer
      const Dtype alpha = sign * loss_weight_ / (*bottom)[0]->num();
      caffe_cpu_axpby(
          (*bottom)[0]->count(),              // count
          alpha,                           // alpha
          diff_.cpu_data(),                // x
          Dtype(0),                        // beta
          (*bottom)[0]->mutable_cpu_diff());  // y
    }
  }
  
}

INSTANTIATE_CLASS(SmoothL1LossLayer);

}  // namespace caffe
