

TF_INC="/home/jhosang/env/py3.4-tensorflow/lib/python3.4/site-packages/tensorflow/include"

.PHONY: all

all: nms_net/det_matching.so nms_net/roi_pooling_layer/roi_pooling.so

nms_net/roi_pooling_layer/roi_pooling.so: nms_net/roi_pooling_layer/roi_pooling_op.o nms_net/roi_pooling_layer/roi_pooling_op_gpu.o
	g++ -std=c++11 -shared $< -o $@ -fPIC -O2

%.o: %.cc
	g++ -std=c++11 -c $< -o $@ -fPIC -I ${TF_INC} -O2

%.o: %.cu
	nvcc -std=c++11 -c $< -o $@ -I ${TF_INC} -O2 -x cu -arch=sm_37

%.so: %.cc
	g++ -std=c++11 -shared $< -o $@ -fPIC -I ${TF_INC} -O2

