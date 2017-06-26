rm -R snapshot
mkdir snapshot

GLOG_logtostderr=1 ../../../C3D-v1.0/C3D_overlap_loss/build/tools/finetune_net.bin solver.prototxt ../network_classification/snapshot/SCNN_uniform16_cls20_iter_30000 2> train.log
