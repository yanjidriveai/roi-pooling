TF_INC = $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_LIB=$(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDA_LIB = /usr/local/cuda-10.0/targets/x86_64-linux/lib/

all: clean build test

build: roi_pooling.so

roi_pooling.cu.o: roi_pooling.cu.cc
	nvcc -std=c++11 -c -o $@ $? -I $(TF_INC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

roi_pooling.so: roi_pooling.cc roi_pooling.cu.o
	g++ -std=c++11 -shared -o $@ $? -I $(TF_INC) -fPIC -L$(TF_LIB) -ltensorflow_framework

test: build
	python roi_pooling_test.py

clean:
	rm -f *.o *.so *.pyc *.npy
