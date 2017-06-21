#include "svm.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include "cusparse.h"

#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

void libsvm2CSR(struct svm_node **xs, const int l, int &nnz, double **valX, int **rowX, int **colX)
{
	std::vector<double> vals;
	std::vector<int> rows, cols;
	vals.reserve(l * 1000);
	rows.reserve(l+1);
	cols.reserve(l * 1000);

	int offset = 0;
	for (int i = 0; i < l; i++) {
		rows.push_back(offset);
		svm_node *x = xs[i];
		for (; x->index != -1; ++x) {
			vals.push_back(x->value);
			cols.push_back(x->index);
			++offset;
		}
	}
	rows.push_back(offset);
	nnz = offset;

	cudaMalloc(valX, sizeof(double)*vals.size());
	cudaMalloc(rowX, sizeof(int)*rows.size());
	cudaMalloc(colX, sizeof(int)*cols.size());
	cudaMemcpy(*valX, vals.data(), sizeof(double)*vals.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(*rowX, rows.data(), sizeof(int)*rows.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(*colX, cols.data(), sizeof(int)*cols.size(), cudaMemcpyHostToDevice);
}

__global__ void rbf_kernel(double gamma, int bsize, int msize,
	double *valB, int *rowB, int *colB,
	double *valM, int *rowM, int *colM,
	double *output)
{
	int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	int idxy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idxx < bsize && idxy < msize) {
		int stx = rowB[idxx], edx = rowB[idxx+1];
		int sty = rowM[idxy], edy = rowM[idxy+1];
		double sum = 0;
		while (stx < edx && sty < edy) {
			if (colB[stx] < colM[sty]) {
				sum += valB[stx] * valB[stx];
				stx++;
			}
			else if (colB[stx] > colM[sty]) {
				sum += valM[sty] * valM[sty];
				sty++;
			}
			else {
				double d = valB[stx] - valM[sty];
				sum += d * d;
				++stx;
				++sty;
			}
		}
		while (stx < edx) {
			sum += valB[stx] * valB[stx];
			stx++;
		}
		while (sty < edy) {
			sum += valM[sty] * valM[sty];
			sty++;
		}
		output[idxy+idxx*msize] = exp(-gamma*sum);
	}
}

__global__ void extend(double *dist, int ncluster, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < ncluster*size) {
		dist[idx+ncluster] = dist[idx%ncluster];
	}
}

__global__ void getIndex(double *dist, int ncluster, int size, int *label)
{
	int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if (idx < size) {
		int st = idx * ncluster;
		double mn = dist[st];
		int mi = 0;
		for (int i = 1; i < ncluster; i++) {
			if (dist[st+i] < mn) {
				mn = dist[st+i];
				mi = i;
			}
		}
		label[idx] = mi;
	}
}

void knkmeans_predict_alllevel(const svm_parameter *param, const svm_problem *prob, struct svm_node** sample,
	int **sub_cidx, int **sub_csize, int msize, double **cluster_avg, int **same_cluster_map,
	const int lvl, const int nchild, int **full_cidx, int **full_csize)
{
	for (int i = 1; i < lvl; i++) {
		for (int j = 0 ; j < (int)pow(nchild,i) ; j++) {
			printf("%d ", sub_csize[i][j]);
		}
		printf("\n");
	}
	const int chunk_size = 10000;

	cusparseHandle_t handle;
	cusparseCreate(&handle);
	cusparseMatDescr_t descr;
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	int maxcluster = (int)pow(nchild, lvl-1);
	double **valC = Malloc(double*, lvl);
	int **rowC = Malloc(int*, lvl), **colC = Malloc(int*, lvl);
	double *tmpval = Malloc(double, msize);
	int *tmprow = Malloc(int, maxcluster);
	for (int i = 1; i < lvl; i++) {
		int ncluster = (int)pow(nchild, i);
		cudaMalloc(&valC[i], sizeof(double)*msize);
		cudaMalloc(&rowC[i], sizeof(int)*(ncluster+1));
		cudaMalloc(&colC[i], sizeof(int)*msize);
		int offset = 0;
		for (int j = 0; j < ncluster; j++) {
			tmprow[j] = offset;
			for (int k = 0; k < sub_csize[i][j]; k++)
				tmpval[offset+k] = 1.0 / sub_csize[i][j];
			offset += sub_csize[i][j];
		}
		tmprow[ncluster] = offset;
		cudaMemcpy(valC[i], tmpval, sizeof(double)*msize, cudaMemcpyHostToDevice);
		cudaMemcpy(rowC[i], tmprow, sizeof(int)*(ncluster+1), cudaMemcpyHostToDevice);
		cudaMemcpy(colC[i], sub_cidx[i], sizeof(int)*msize, cudaMemcpyHostToDevice);
	}
	free(tmpval); free(tmprow);

	double *valS;
	int *rowS, *colS, nnzS;
	libsvm2CSR(sample, msize, nnzS, &valS, &rowS, &colS); 

	double *dist;
	cudaMalloc(&dist, sizeof(double)*maxcluster*chunk_size);

	int **label = Malloc(int*, lvl);
	for (int i = 0; i < lvl; i++)
		label[i] = Malloc(int, prob->l);

	for (int i = 0; i < prob->l; i++)
		full_cidx[0][i] = i;
	full_csize[0][0] = prob->l;
	
	double *K;
	cudaMalloc(&K, sizeof(double)*chunk_size*msize);
			
	int *clabel;
	cudaMalloc(&clabel, sizeof(int)*chunk_size);

	for (int i = 0; i < prob->l; i += chunk_size) {
		int sz = std::min(chunk_size, prob->l - i);
		double *valX;
		int *rowX, *colX, nnzX;
		libsvm2CSR(&prob->x[i], sz, nnzX, &valX, &rowX, &colX);
		dim3 gdim(CeilDiv(sz,32), CeilDiv(msize,16)), bdim(32,16);
		rbf_kernel<<<gdim, bdim>>>(param->gamma, sz, msize,
				valX, rowX, colX, valS, rowS, colS, K);
		cudaFree(valX);
		cudaFree(rowX);
		cudaFree(colX);

		for (int l = 1; l < lvl; l++) {
			int ncluster = (int)pow(nchild, l);
			cudaMemcpy(dist, cluster_avg[l], sizeof(double)*ncluster, cudaMemcpyHostToDevice);
			extend<<<CeilDiv(ncluster*(sz-1), 256), 256>>>(dist, ncluster, sz-1);
			const double dtwo = -2, done = 1;
			cusparseDcsrmm(handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				ncluster, sz, msize, msize,
				&dtwo, descr,
				valC[l], rowC[l], colC[l], K, msize, &done, dist, ncluster);
			getIndex<<<CeilDiv(sz, 256), 256>>>(dist, ncluster, sz, clabel);
			cudaMemcpy(&label[l][i], clabel, sizeof(int)*sz, cudaMemcpyDeviceToHost);
		}
	}
	cudaFree(valS); cudaFree(rowS); cudaFree(colS);
	cudaFree(dist); cudaFree(K); cudaFree(clabel);
	for (int i = 1; i < lvl; i++) {
		cudaFree(valC[i]); cudaFree(rowC[i]); cudaFree(colC[i]);
	}
	free(valC); free(rowC); free(colC);
		
	for (int l = 2; l < lvl; l++) {
		for (int i = 0; i < prob->l; i++) {
			const int cid = same_cluster_map[l-1][label[l-1][i]];
			if (cid > 0) {
				label[l-1][i] = -1;
				label[l][i] = cid;
			}
		}
	}
	
	int *cur_start = Malloc(int, (int)pow(nchild, lvl-1));
	for (int l = 1; l < lvl; l++) {
		int ncluster = (int)pow(nchild, l);
		for (int i = 0; i < ncluster; i++)
			full_csize[l][i] = 0;
		for (int i = 0; i < prob->l; i++) {
			if (label[l][i] < 0) continue;
			full_csize[l][label[l][i]]++;
		}
		int sum = 0;
		for (int i = 0; i < ncluster; i++) {
			cur_start[i] = sum;
			sum += full_csize[l][i];
		}
		for (int i = 0; i < prob->l; i++) {
			if (label[l][i] < 0) continue;
			int &idx = cur_start[label[l][i]];
			full_cidx[l][idx] = i;
			idx++;
		}
	}

	for (int i = 0; i < lvl; i++) free(label[i]);
	free(label);
	free(cur_start);
}

