#ifndef __CUDA_KMEANS_H__
#define __CUDA_KMEANS_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void distanceWithCuda(double* points, int numOfPoints, double* clustersCenters, int numOfClusters, int dimension, double* results, int pointsPerBlock);
__global__ void findClosestCluster(double* distances, int numOfClusters, int numOfPoints, int pointsPerBlock, int* belongTo);
cudaError_t dividePointToClustersWithCuda(double* pointsCordinationsPointer, int numOfPoints, double* clustersCentersCordinations, int numOfClusters, int dimension, int** belongTo);
void handleErrors(cudaError_t cudaStatus, const char* errorMessage);
cudaError_t copyPointsCordToCUDA(double* pointsCordinations, int numOfPoints, int dimension, double** pointsCordCUDAPointer);

#endif // !__CUDA_KMEANS_H__




