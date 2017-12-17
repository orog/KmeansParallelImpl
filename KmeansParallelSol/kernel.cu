#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void distanceWithCuda(double* points, int numOfPoints, double* clustersCenters, int numOfClusters, int dimension, double* results, int pointsPerBlock)
{
	int i, blockID = blockIdx.x;
	double result = 0;

	if (blockID == gridDim.x - 1 && numOfPoints % blockDim.x <= threadIdx.x)
		return;

	for (i = 0; i < dimension; i++)
	{
		result += (points[(blockID*pointsPerBlock + threadIdx.x)*dimension + i] - clustersCenters[threadIdx.y*dimension + i]) *  (points[(blockID*pointsPerBlock + threadIdx.x)*dimension + i] - clustersCenters[threadIdx.y*dimension + i]);
	}
	results[numOfPoints*threadIdx.y + (blockID*pointsPerBlock + threadIdx.x)] = result;
}

__global__ void findClosestCluster(double* distances, int numOfClusters, int numOfPoints, int pointsPerBlock, int* belongTo)
{
	int i, xid = threadIdx.x, blockId = blockIdx.x;
	double minIndex = 0, minDistance, tempDistance;

	if (blockIdx.x == gridDim.x - 1 && numOfPoints % blockDim.x <= xid)
		return;

	minDistance = distances[pointsPerBlock*blockId + xid];

	for (i = 1; i < numOfClusters; i++)
	{
		tempDistance = distances[pointsPerBlock*blockId + xid + i*numOfPoints];
		if (minDistance > tempDistance)
		{
			minIndex = i;
			minDistance = tempDistance;
		}
	}
	belongTo[pointsPerBlock*blockId + xid] = minIndex;
}

cudaError_t dividePointToClustersWithCuda(double* pointsCoordinationsPointer, int numOfPoints, double* clustersCentersCoordinations, int numOfClusters, int dimension, int** belongTo)
{
	double *dev_clustersCentersCoordinations = NULL;
	double *dev_distances = NULL;
	int pointsPerBlock, numBlocks, i, extraBlock , *dev_belongTo = NULL;
	cudaDeviceProp prop;
	cudaError_t cudaStatus;

	*belongTo = (int*)calloc(numOfPoints, sizeof(int));

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	handleErrors(cudaStatus, "cudaSetDevice failed!");

	cudaStatus = cudaGetDeviceProperties(&prop, 0);
	handleErrors(cudaStatus, "cudaGetDeviceProperties failed!");

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_clustersCentersCoordinations, dimension * numOfClusters * sizeof(double));
	handleErrors(cudaStatus, "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_distances, numOfPoints * numOfClusters * sizeof(double));
	handleErrors(cudaStatus, "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_belongTo, numOfPoints * sizeof(int));
	handleErrors(cudaStatus, "cudaMalloc failed!");

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_clustersCentersCoordinations, clustersCentersCoordinations, dimension * numOfClusters * sizeof(double), cudaMemcpyHostToDevice);
	handleErrors(cudaStatus, "cudaMemcpy failed!");


	// Launch a kernel on the GPU with one thread for each element.
	pointsPerBlock = prop.maxThreadsPerBlock / numOfClusters;
	dim3 dim(pointsPerBlock, numOfClusters);
	numBlocks = numOfPoints / pointsPerBlock;
	if (numOfPoints % pointsPerBlock == 0)
		extraBlock = 0;
	else
		extraBlock = 1;
	distanceWithCuda << <numBlocks + extraBlock, dim >> > (pointsCoordinationsPointer, numOfPoints, dev_clustersCentersCoordinations, numOfClusters, dimension, dev_distances, pointsPerBlock);

	cudaStatus = cudaDeviceSynchronize();
	handleErrors(cudaStatus, "cudaDeviceSynchronize1 failed!\n");

	pointsPerBlock = prop.maxThreadsPerBlock;
	numBlocks = numOfPoints / pointsPerBlock;
	if (numOfPoints % pointsPerBlock == 0)
		extraBlock = 0;
	else
		extraBlock = 1;
	findClosestCluster << <numBlocks + extraBlock, pointsPerBlock >> > (dev_distances, numOfClusters, numOfPoints, pointsPerBlock, dev_belongTo);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	handleErrors(cudaStatus, "addKernel launch failed: %s\n");


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	handleErrors(cudaStatus, "cudaDeviceSynchronize2 failed!\n");



	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(*belongTo, dev_belongTo, numOfPoints * sizeof(int), cudaMemcpyDeviceToHost);
	handleErrors(cudaStatus, "cudaMemcpy failed!");
	//cudaFree(dev_pointsCordinations);
	cudaFree(dev_clustersCentersCoordinations);
	cudaFree(dev_distances);
	cudaFree(dev_belongTo);
	return cudaStatus;
}

cudaError_t copyPointsCordToCUDA(double* pointsCoordinations, int numOfPoints,int dimension , double** pointsCoordCUDAPointer)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	handleErrors(cudaStatus, "copyPointsCord cudaSetDevice failed!\n");

	cudaStatus = cudaMalloc((void**)pointsCoordCUDAPointer, numOfPoints*dimension * sizeof(double));
	handleErrors(cudaStatus, "copyPointsCord cudaMalloc failed!\n");
	cudaStatus = cudaMemcpy(*pointsCoordCUDAPointer, pointsCoordinations, numOfPoints*dimension * sizeof(double), cudaMemcpyHostToDevice);
	handleErrors(cudaStatus, "copyPointsCord cudaMemcpy failed!\n");
	return cudaStatus;
}



void handleErrors(cudaError_t cudaStatus, const char* errorMessage)
{
	if (cudaStatus != cudaSuccess)
	{
		printf(errorMessage);
		fflush(stdout);
		system("pause");
		exit(1);
	}
}
