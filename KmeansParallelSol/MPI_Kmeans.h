#ifndef __MPI_KMEANS_H__
#define __MPI_KMEANS_H__
#include <mpi.h>
#include "point.h"
#include "cluster.h"
#include "Kmean_info.h"
#include <stdio.h>
#include <stdlib.h>

void intialMPI(int* myid, int* numprocs, int* argc, char*** argv);

void createPointDatatype(MPI_Datatype* pointDatatype);

void createClusterDatatype(MPI_Datatype* clusterDatatype);

void createInfoDatatype(MPI_Datatype* infoDatatype);

void scattervDatabase(point_t* pointsDatabase, double* coordsDatabase, int rootDatabaseSize, int othersDatabaseSize, int root, int pointDim, MPI_Datatype pointDatatype, point_t** myPoints, double** myCoords);

void gathervBelongTo(point_t* pointsDatabase, int totalNumOfPoints, point_t* myPoints, int rootDatabaseSize, int salveDatabaseSize, MPI_Datatype pointDatatype, int root);

void bCastCenters(cluster_t** clusters, double** clusterCoords, int numOfClusters, int coordDim, MPI_Datatype clusterDatatype, int root);

void synchronizedCenters(cluster_t* clusters, int numOfClusters);

void ParallelKmeansAlg(cluster_t* clusters, double* clustersCords, int numOfClusters, point_t* myPoints, double* pointsCords, double* pointsCordsCUDAPointer, int numOfPoints, int dimension, int maxNumOfIter);


#endif // !__MPI_KMEANS_H__




