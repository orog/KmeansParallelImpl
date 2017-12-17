#ifndef __CLUSTER_H__
#define __CLUSTER_H__
#include "point.h"
struct MyCluster
{
	int numOfPoints;
	int numOfPointsInProc;
	int id;
	int dimension;
	double* center;
	double* sumCenter;
}typedef cluster_t;

void intialClusters(cluster_t** clusters, double** clusterCoords, int pointDim, int intialNumOfClusters, point_t* points);

void intialCluster(cluster_t* cluster, point_t* center, int id, double* coordsPointer);

#endif // !__CLUSTER_H__
