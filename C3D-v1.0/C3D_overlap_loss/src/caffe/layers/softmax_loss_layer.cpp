// Copyright 2014 BVLC and contributors.

/* ----------------------------------------------------------------------------------------------------------------
 * Segment-CNN
 * Copyright (c) 2016 , Digital Video & Multimedia (DVMM) Laboratory at Columbia University in the City of New York.
 * Licensed under The MIT License [see LICENSE for details]
 * Written by Zheng Shou, Dongang Wang, and Shih-Fu Chang.
---------------------------------------------------------------------------------------------------------------- */

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 3) << "SoftmaxLoss Layer takes three blobs as input.";	// 100515
  CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* overlap = bottom[2]->cpu_data();	// 100515
  int num = prob_.num();
  int dim = prob_.count() / num;
  Dtype loss_softmax = 0;	// 100515
  Dtype loss_overlap = 0;	// 100515
  Dtype loss = 0;	// 100515
  float lambda = 1; // 100515
  for (int i = 0; i < num; ++i) {
    loss_softmax -= log(max(prob_data[i * dim + static_cast<int>(label[i])],
                     Dtype(FLT_MIN)));
	if((static_cast<int>(label[i]))!=0) {
		
		loss_overlap += 0.5*(prob_data[i * dim + static_cast<int>(label[i])]/static_cast<float>(std::sqrt(std::sqrt(std::sqrt(overlap[i])))))*(prob_data[i * dim + static_cast<int>(label[i])]/static_cast<float>(std::sqrt(std::sqrt(std::sqrt(overlap[i]))))) - 0.5;	// 100515
	}
  }
  LOG(INFO) << "softmax loss: " << loss_softmax/num;	// 100515
  LOG(INFO) << "overlap loss: " << lambda*loss_overlap/num;	// 100515
  loss = loss_softmax + lambda*loss_overlap;	// 100515
  LOG(INFO) << "total loss: " << loss/num;
  return loss / num;
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prob_data = prob_.cpu_data();
  memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
  const Dtype* label = (*bottom)[1]->cpu_data();
  const Dtype* overlap = (*bottom)[2]->cpu_data();	// 100515
  int num = prob_.num();
  int dim = prob_.count() / num;
  float lambda = 1; // 100515
  
  // 100515
  for (int i = 0; i < num; ++i) {	// data
	  for (int j = 0; j < dim; ++j) {	
		  if((static_cast<int>(label[i]))!=j) {
			  if((static_cast<int>(label[i]))!=0) {
					bottom_diff[i * dim + j] -= lambda*(prob_data[i * dim + j]*prob_data[i * dim + static_cast<int>(label[i])]*prob_data[i * dim + static_cast<int>(label[i])])/(static_cast<float>(std::sqrt(std::sqrt(std::sqrt(overlap[i]))))*static_cast<float>(std::sqrt(std::sqrt(std::sqrt(overlap[i])))));
			  }
		  } else {
			  bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
			  if((static_cast<int>(label[i]))!=0) {
					bottom_diff[i * dim + j] += lambda*(1-prob_data[i * dim + static_cast<int>(label[i])])*(prob_data[i * dim + static_cast<int>(label[i])]*prob_data[i * dim + static_cast<int>(label[i])])/(static_cast<float>(std::sqrt(std::sqrt(std::sqrt(overlap[i]))))*static_cast<float>(std::sqrt(std::sqrt(std::sqrt(overlap[i])))));	
			  }
			}
		}
	}
////////////////////////////////////////////////////////////////////

  // Scale down gradient
  caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
}


INSTANTIATE_CLASS(SoftmaxWithLossLayer);


}  // namespace caffe
