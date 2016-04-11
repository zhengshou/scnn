// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 3) << "Smooth_L1_Loss Layer takes three blobs as input.";
  CHECK_LE(top->size(), 1) << "Loss Layer takes no more than one output.";
  if (top->size() == 1) {
   // Layers should copy the loss in the top blob
   (*top)[0]->Reshape(1, 1, 1, 1, 1);
  }
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and box target should have the same number.";
  CHECK_EQ(bottom[0]->num(), bottom[2]->num())
      << "The data and box loss weight should have the same number.";
  FurtherSetUp(bottom, top);
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::FurtherSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  
  has_weights_ = (bottom.size() == 3);	// set has_weights_
  loss_weight_ = 1; // loss_weight_ for smooth loss layer
  
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1,
      bottom[0]->height(), bottom[0]->width());
}


template <typename Dtype>
Dtype SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS(SmoothL1LossLayer);

}  // namespace caffe
