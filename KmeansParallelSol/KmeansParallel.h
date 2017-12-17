
#ifndef __KMEANS_PARALLEL_H__
#define __KMEANS_PARALLEL_H__
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include<string.h>
#include <omp.h>
#include <mpi.h>
#include "cluster.h"
#include "point.h"
#include "Kmean_info.h"
#include "kernel.h"

void output_file(char* outputPath, cluster_t* clusters, int numOfClusters, double quality);

void read_file(char* dataSetPath, int* numOfPoints, int* maxNumOfClusters, int* maxNumOfIter, double* qualityMeasure, int* dimension, point_t** points, double** coords);

double distance(double* point1, double* point2, int size);

void calculateSumCenters(point_t* points, int numOfPoints, cluster_t* clusters, int numOfClusters);

void calculateCenters(point_t* points, int numOfPoints, cluster_t* clusters, int numOfClusters);

double calcQuality(cluster_t* clusters, int numOfClusters, point_t* pointDatabase, int numOfPoints, int dimension);

double* getDiameters(point_t* points, int numOfPoints, int numOfClusters, int dimension);

void dividePointToClusters(cluster_t* clusters, double* clustersCords, int numOfClusters, point_t* points, double* pointsCordsCUDAPointer, int numOfPoints, int dimension, int* change);

void dividePointToClustersWithOmp(cluster_t* clusters, double* clustersCords, int numOfClusters, point_t* points, double* pointsCords, int numOfPoints, int dimension, int* change);

#endif // !__KMEANS_PARALLEL_H__