# libsvm-gg

LIBSVM with GPU Goodness: integrating
[DCSVM](http://www.cs.utexas.edu/~cjhsieh/dcsvm/) and
[GTSVM](http://ttic.uchicago.edu/~cotter/projects/gtsvm/). See
[details](#details) section.

## Usage

In `libsvm-3.22` directory, use `make` to build the GTSVM library and LIBSVM
binaries at once.

Extra flags compared to LIBSVM:

- `-G`: enable GTSVM backend
- `-eze [1-5]`: specify clustering levels in DCSVM

## Details

[LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) [1] is a popular SVM
library used in various machine learning applications. However its CPU-based
solver cannot handle large datasets in reasonable time. In `libsvm-gg` we
utilize the GPU for model training/prediction by integrating two techniques:
DCSVM [2] and GTSVM [3].

[DCSVM](http://www.cs.utexas.edu/~cjhsieh/dcsvm/) is a divide-and-conquer solver
for kernel SVMs. With the dataset partitioned into sub-problems through kernel
clustering, the authors show that overall training speed can be significantly
improved by solving each sub-problem independently, and conbine local solutions
to initialize the original problem. A reference implementation is provided by
the authors, written in MATLAB and uses a modified LIBSVM to produce unbiased
models. We implement the DC logic in LIBSVM directly, and use a CUDA
implementation to accelerate kernel k-means clustering.

[GTSVM](http://ttic.uchicago.edu/~cotter/projects/gtsvm/) is a kernel SVM solver
implemented in CUDA, providing promising performance. However its interface is
not familiar to LIBSVM users, who may have a hard time switching to GTSVM for
the performance gain in their existing machine learning systems. We replace the
underlying C-SVC solver of LIBSVM into GTSVM, enabling users to benefit from
GTSVM while interfaces remain compatible.

This project is part of the GPU Programming course given at Dept. of Computer
Science and Engineering, National Taiwan University.

## References

[1] Chang, C. C., & Lin, C. J. (2011). LIBSVM: a library for support vector
machines. ACM transactions on intelligent systems and technology (TIST), 2(3),
27.

[2] Hsieh, C. J., Si, S., & Dhillon, I. (2014, January). A
divide-and-conquer solver for kernel support vector machines. In
International Conference on Machine Learning (pp. 566-574).

[3] Cotter, A., Srebro, N., & Keshet, J. (2011, August). A GPU-tailored
approach for training kernelized SVMs. In Proceedings of the 17th ACM SIGKDD
international conference on Knowledge discovery and data mining (pp.
805-813). ACM.
