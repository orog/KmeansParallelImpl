#include "cluster.h"
#include <stdlib.h>
#include <string.h>

void intialClusters(cluster_t** clusters, double** clusterCoords, int pointDim, int intialNumOfClusters, point_t* points)
{
	int i;
	*clusters = (cluster_t*)calloc(intialNumOfClusters, sizeof(cluster_t));
	*clusterCoords = (double*)calloc(intialNumOfClusters * pointDim, sizeof(double));
	for (i = 0; i < intialNumOfClusters; i++)
	{
		intialCluster(&(*clusters)[i], &points[i], i, &((*clusterCoords)[i * pointDim]));
	}
}

void intialCluster(cluster_t* cluster, point_t* center, int id, double* coordsPointer)
{
	cluster->id = id;
	cluster->numOfPoints = 0;
	cluster->numOfPointsInProc = 0;
	cluster->dimension = center->dimension;
	cluster->center = coordsPointer;
	cluster->sumCenter = (double*)calloc(center->dimension, sizeof(double));
	memcpy(cluster->center, center->coordinations, sizeof(double)*center->dimension);
}

