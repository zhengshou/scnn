rm -R snapshot
mkdir snapshot

GLOG_logtostderr=1 ../../../C3D_sample_rate/build/tools/finetune_net.bin solver.prototxt ../../../models/conv3d_deepnetA_sport1m_iter_1900000 2> train.log
