#define CHANGE_TAG 1
#define NO_CHANGE_TAG 0
#include "MPI_Kmeans.h"
#include "KmeansParallel.h"


void intialMPI(int* myid, int* numprocs, int* argc, char*** argv)
{
	MPI_Init(argc, argv);
	MPI_Comm_rank(MPI_COMM_WORLD, myid);
	MPI_Comm_size(MPI_COMM_WORLD, numprocs);
}

void createPointDatatype(MPI_Datatype* pointDatatype)
{
	point_t point;
	MPI_Datatype type[4] = { MPI_INT ,MPI_INT ,MPI_INT ,MPI_DOUBLE };
	int blocklen[4] = { 1,1,1,1 };
	MPI_Aint disp[4];
	disp[0] = (char *)&point.dimension - (char *)&point;
	disp[1] = (char *)&point.id - (char *)&point;
	disp[2] = (char *)&point.belongTo - (char *)&point;
	disp[3] = (char*)&point.coordinations - (char*)&point;
	MPI_Type_create_struct(4, blocklen, disp, type, pointDatatype);
	MPI_Type_commit(pointDatatype);
}

void createClusterDatatype(MPI_Datatype* clusterDatatype)
{
	cluster_t cluster;
	MPI_Datatype type[6] = { MPI_INT,MPI_INT,MPI_INT ,MPI_INT,MPI_DOUBLE ,MPI_DOUBLE};
	int blocklen[6] = {1, 1,1,1,1 ,1};
	MPI_Aint disp[6];
	disp[0] = (char*)&cluster.numOfPoints - (char*)&cluster;
	disp[1] = (char*)&cluster.numOfPointsInProc - (char*)&cluster;
	disp[2] = (char*)&cluster.id - (char*)&cluster;
	disp[3] = (char*)&cluster.dimension - (char*)&cluster;
	disp[4] = (char*)&cluster.center - (char*)&cluster;
	disp[5] = (char*)&cluster.sumCenter - (char*)&cluster;
	MPI_Type_create_struct(6, blocklen, disp, type, clusterDatatype);
	MPI_Type_commit(clusterDatatype);
}

void createInfoDatatype(MPI_Datatype* infoDatatype)
{
	kmean_info_t info;
	MPI_Datatype type[8] = { MPI_INT,MPI_INT ,MPI_INT ,MPI_INT,MPI_INT,MPI_INT ,MPI_DOUBLE,MPI_DOUBLE };
	int blocklen[8] = { 1,1,1,1,1,1,1,1};
	MPI_Aint disp[8];
	disp[0] = (char*)&info.numOfPoints - (char*)&info;
	disp[1] = (char*)&info.maxNumOfClusters - (char*)&info;
	disp[2] = (char*)&info.maxNumOfIter - (char*)&info;
	disp[3] = (char*)&info.currentNumOfClusters - (char*)&info;
	disp[4] = (char*)&info.salveDatabaseSize - (char*)&info;
	disp[5] = (char*)&info.dimension - (char*)&info;
	disp[6] = (char*)&info.qualityMeasure - (char*)&info;
	disp[7] = (char*)&info.currentQuality - (char*)&info;
	MPI_Type_create_struct(8, blocklen, disp, type, infoDatatype);
	MPI_Type_commit(infoDatatype);
}

void scattervDatabase(point_t* pointsDatabase,double* coordsDatabase,int rootDatabaseSize,int othersDatabaseSize, int root, int pointDim,MPI_Datatype pointDatatype,point_t** myPoints,double** myCoords)
{
	int i,procID, numOfProcs, *countsPoints, *dispPoints, *countsCoords, *dispCoords, counter = 0,myDatabaseSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcs);

	countsPoints = (int*)calloc(numOfProcs, sizeof(int));
	countsCoords = (int*)calloc(numOfProcs, sizeof(int));
	dispPoints = (int*)calloc(numOfProcs, sizeof(int));
	dispCoords = (int*)calloc(numOfProcs, sizeof(int));

	if (procID == root)
	{
		myDatabaseSize = rootDatabaseSize;
		for (i = 0; i < numOfProcs; i++)
		{
			dispCoords[i] = counter * pointDim;
			dispPoints[i] = counter;
			if (i == root)
			{
				countsPoints[i] = rootDatabaseSize;
				countsCoords[i] = rootDatabaseSize * pointDim;
				counter += rootDatabaseSize;
			}
			else
			{
				countsPoints[i] = othersDatabaseSize;
				countsCoords[i] = othersDatabaseSize * pointDim;
				counter += othersDatabaseSize;
			}
		}
	}
	else
		myDatabaseSize = othersDatabaseSize;

	*myPoints = (point_t*)calloc(myDatabaseSize, sizeof(point_t));
	*myCoords = (double*)calloc(myDatabaseSize * pointDim, sizeof(double));
	
	MPI_Scatterv(pointsDatabase, countsPoints, dispPoints, pointDatatype, *myPoints, myDatabaseSize, pointDatatype, root, MPI_COMM_WORLD);
	MPI_Scatterv(coordsDatabase, countsCoords, dispCoords, MPI_DOUBLE, *myCoords, myDatabaseSize * pointDim, MPI_DOUBLE, root, MPI_COMM_WORLD);

#pragma omp parallel for	
	for (i = 0; i < myDatabaseSize; i++)
		(*myPoints)[i].coordinations = &((*myCoords)[i * pointDim]);

	free(countsCoords);
	free(countsPoints);
	free(dispPoints);
	free(dispCoords);
}

void bCastCenters(cluster_t** clusters,double** clusterCoords, int numOfClusters,int coordDim, MPI_Datatype clusterDatatype,int root)
{
	int i, procID;
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);
	if (procID != root)
	{
		*clusterCoords = (double*)calloc(numOfClusters * coordDim, sizeof(double));
		*clusters = (cluster_t*)calloc(numOfClusters, sizeof(cluster_t));
	}
	MPI_Bcast(*clusters, numOfClusters, clusterDatatype, root, MPI_COMM_WORLD);
	MPI_Bcast(*clusterCoords,numOfClusters * coordDim,MPI_DOUBLE,root,MPI_COMM_WORLD);

	for (i = 0; i < numOfClusters && (procID != root); i++)
	{
		(*clusters)[i].center = &((*clusterCoords)[i * coordDim]);
		(*clusters)[i].sumCenter = (double*)calloc((*clusters)[i].dimension, sizeof(double));
	}
}

void synchronizedCenters(cluster_t* clusters, int numOfClusters)
{
	int i, j,numOfPointsInCluster;
	double* newCenter = (double*)calloc(clusters[0].dimension, sizeof(double));
	for (i = 0; i < numOfClusters; i++)
	{
		MPI_Allreduce(&(clusters[i].numOfPointsInProc), &numOfPointsInCluster, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(clusters[i].sumCenter, newCenter, clusters[i].dimension, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#pragma omp parallel for
		for (j = 0; j < clusters[i].dimension; j++)
		{
			clusters[i].center[j] = (newCenter[j] / numOfPointsInCluster);
			clusters[i].sumCenter[j] = 0;
			newCenter[j] = 0;
		}
		clusters[i].numOfPoints = numOfPointsInCluster;
		numOfPointsInCluster = 0;
	}
	free(newCenter);
}

void ParallelKmeansAlg(cluster_t* clusters, double* clustersCoords, int numOfClusters, point_t* myPoints, double* pointsCoords,double* pointsCoordsCUDAPointer, int numOfPoints, int dimension,int maxNumOfIter)
{
	int change = CHANGE_TAG, tempChange = NO_CHANGE_TAG, iterPerformed = 1,OMP_or_CUDA = 0,i;
	int procID;
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);
	//Group points around the given cluster centers
	for (i = 0; i < numOfPoints; i++)
		myPoints[i].belongTo = -1;
	
	if(OMP_or_CUDA)
		dividePointToClustersWithOmp(clusters, clustersCoords, numOfClusters, myPoints, pointsCoords, numOfPoints, dimension, &change);
	else
		dividePointToClusters(clusters, clustersCoords, numOfClusters, myPoints, pointsCoordsCUDAPointer, numOfPoints, dimension, &change);
	
	calculateCenters(myPoints, numOfPoints, clusters, numOfClusters);
	//change = CHANGE_TAG;
	while (iterPerformed < maxNumOfIter && change)
	{
		tempChange = NO_CHANGE_TAG;
		change = NO_CHANGE_TAG;
		if(OMP_or_CUDA)
			dividePointToClustersWithOmp(clusters, clustersCoords, numOfClusters, myPoints, pointsCoords, numOfPoints, dimension, &tempChange);
		else
			dividePointToClusters(clusters, clustersCoords, numOfClusters, myPoints, pointsCoordsCUDAPointer, numOfPoints, dimension, &tempChange);
		MPI_Allreduce(&tempChange, &change, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		if (change != 0)
			calculateCenters(myPoints, numOfPoints, clusters, numOfClusters);
		iterPerformed++;

	}
}

void gathervBelongTo(point_t* pointsDatabase, int totalNumOfPoints,point_t* myPoints, int rootDatabaseSize, int salveDatabaseSize ,MPI_Datatype pointDatatype,int root)
{
	int* disp = NULL, *counts = NULL,procID, numOfProc,counter = 0, i, myDatabaseSize,idx =810;
	point_t* totalPoints = NULL;
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);
	
	if (procID == root)
	{
		totalPoints = (point_t*)calloc(totalNumOfPoints , sizeof(point_t));
		disp = (int*)calloc(numOfProc, sizeof(int));
		counts = (int*)calloc(numOfProc, sizeof(int));
		myDatabaseSize = rootDatabaseSize;
		for (i = 0; i < numOfProc; i++)
		{
			disp[i] = counter;
			if (i == root)
			{
				counts[i] = rootDatabaseSize;
				counter += rootDatabaseSize;
			}
			else 
			{
				counts[i] = salveDatabaseSize;
				counter += salveDatabaseSize;
			}
		}
	}
	else
	{
		myDatabaseSize = salveDatabaseSize;
	}
	MPI_Gatherv(myPoints, myDatabaseSize, pointDatatype, totalPoints, counts, disp, pointDatatype, root, MPI_COMM_WORLD);
	
	if (procID == root)
	{
		for (i = 0; i < totalNumOfPoints; i++)
		{
			pointsDatabase[totalPoints[i].id].belongTo = totalPoints[i].belongTo;
		}
		free(disp);
		free(counts);
	}
}






	


