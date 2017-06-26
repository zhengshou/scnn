# ./finetuning.sh 2>&1 | tee log.train-val

rm -R snapshot
mkdir snapshot

GLOG_logtostderr=1 ../../../C3D-v1.1/C3D_overlap_loss/build/tools/caffe train --solver=solver_r2.prototxt --weights=../network_classification/snapshot/c3d_resnet18_sports1m_r2_iter_14704.caffemodel