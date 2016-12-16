

TF_INC="/home/jhosang/env/tensorflow/local/lib/python2.7/site-packages/tensorflow/include"

.PHONY: all

all: nms_net/det_matching.so

%.so: %.cc
	g++ -std=c++11 -shared $< -o $@ -fPIC -I ${TF_INC} -O2
