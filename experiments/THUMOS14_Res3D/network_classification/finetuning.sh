# ./finetuning.sh 2>&1 | tee log.train-val

rm -R snapshot
mkdir snapshot

GLOG_logtostderr=1 ../../../C3D-v1.1/C3D_sample_rate/build/tools/caffe train --solver=solver_r2.prototxt --weights=../../../models/c3d_resnet18_sports1m_r2_iter_2800000.caffemodel