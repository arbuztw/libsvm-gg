CXX ?= g++
NVCC ?= nvcc
CFLAGS = -Wall -Wconversion -fPIC -g
SHVER = 2
OS = $(shell uname)

all: svm-train svm-predict svm-scale

lib: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} svm.o -o libsvm.so.$(SHVER)

gtsvm:
	$(MAKE) -C ../GTSVM all

svm-predict: svm-predict.c svm.o cusvm.o gtsvm
	$(CXX) $(CFLAGS) svm-predict.c svm.o cusvm.o -o svm-predict -L../GTSVM/lib -lm -lgtsvm -lcudart -lcusparse
svm-train: svm-train.c svm.o cusvm.o gtsvm
	$(CXX) $(CFLAGS) svm-train.c svm.o cusvm.o -o svm-train -L../GTSVM/lib -lm -lgtsvm -lcudart -lcusparse
svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale
svm.o: svm.cpp svm.h
	$(CXX) -I../GTSVM/lib $(CFLAGS) -c svm.cpp
cusvm.o: cusvm.cu svm.h
	$(NVCC) -c cusvm.cu
clean:
	rm -f *~ svm.o svm-train svm-predict svm-scale libsvm.so.$(SHVER)
