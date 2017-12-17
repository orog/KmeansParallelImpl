#include "KmeansParallel.h"
#include "MPI_Kmeans.h"

void output_file(char* outputPath, cluster_t* clusters, int numOfClusters, double quality)
{
	int i, j;
	FILE *f = fopen(outputPath, "w");
	fprintf(f,"\n######  FINAL #####\nK = %d , Q = %lf\n Center of the clusters: \n", numOfClusters, quality);
	for (i = 0; i < numOfClusters; i++)
	{
		fprintf(f,"C%d : \n", i);
		for (j = 0; j < clusters[i].dimension; j++)
			fprintf(f,"%-5.2lf", clusters[i].center[j]);
		fprintf(f,"\n");
	}
	fclose(f);
}

void read_file(char* dataSetPath, int* numOfPoints, int* maxNumOfClusters, int* maxNumOfIter, double* qualityMeasure, int* dimension
	, point_t** points, double** coords)
{
	char id[7];
	int  i, j;
	FILE* f = fopen(dataSetPath, "r");
	if (f == NULL)
		return;
	fscanf(f, "%d", numOfPoints);
	fscanf(f, "%d", dimension);
	fscanf(f, "%d", maxNumOfClusters);
	fscanf(f, "%d", maxNumOfIter);
	fscanf(f, "%lf", qualityMeasure);
	*points = (point_t*)calloc(*numOfPoints, sizeof(point_t));
	*coords = (double*)calloc((*numOfPoints)*(*dimension), sizeof(double));
	for (i = 0; i < *numOfPoints; i++)
	{
		(*points)[i].dimension = *dimension;
		(*points)[i].belongTo = -1;
		(*points)[i].coordinations = &((*coords)[i * (*dimension)]);
		fscanf(f, "%s", id);
		(*points)[i].id = i;
		for (j = 0; j < *dimension; j++)
		{
			fscanf(f, "%lf", &((*points)[i].coordinations[j]));
		}
	}
	fclose(f);
}

double distance(double* point1, double* point2, int size)
{
	int i;
	double distance = 0;
	for (i = 0; i < size; i++)
		distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
	distance = sqrt(distance);
	return distance;

}

void calculateSumCenters(point_t* points, int numOfPoints, cluster_t* clusters, int numOfClusters)
{
	int i, j;
	for (i = 0; i < numOfClusters; i++)
	{
		for (j = 0; j < clusters[i].dimension; j++)
			clusters[i].sumCenter[j] = 0;
	}
	for (i = 0; i < numOfPoints; i++)
	{
#pragma omp parallel for 
		for (j = 0; j < points[i].dimension; j++)
		{
			clusters[(points[i].belongTo)].sumCenter[j] += points[i].coordinations[j];
		}
	}
}

double calcQuality(cluster_t* clusters, int numOfClusters,point_t* pointDatabase,int numOfPoints ,int dimension)
{
	int i, j;
	double quality = 0;
	double *diameterOfclusters = NULL;
	diameterOfclusters = getDiameters(pointDatabase,numOfPoints, numOfClusters, dimension);
#pragma omp parallel for private(j) reduction(+ : quality)
	for (i = 0; i < numOfClusters; i++)
	{
		for (j = i + 1; j < numOfClusters; j++)
		{
			quality += (diameterOfclusters[i] + diameterOfclusters[j]) / distance((clusters[i].center), (clusters[j].center), dimension);
		}
	}
	free(diameterOfclusters);
	return quality / (numOfClusters*(numOfClusters - 1)); //quality / counter;
}

double* getDiameters(point_t* points, int numOfPoints, int numOfClusters,int dimension)
{
	int i, j,threadID,offset,maxThreads = omp_get_max_threads();
	double temp;
	double *diameters = (double*)(calloc(numOfClusters, sizeof(double)));
	double *diametersThreads = (double*)calloc(maxThreads * numOfClusters, sizeof(double));

#pragma omp parallel for private(j,threadID,temp,offset)
	for (i = 0; i < numOfPoints; i++)
	{
		threadID = omp_get_thread_num();
		temp = 0;
		offset = threadID * numOfClusters;
		for (j = i + 1; j < numOfPoints; j++)
		{
			if (points[i].belongTo == points[j].belongTo)
			{
				temp = distance(points[i].coordinations, points[j].coordinations,dimension);

				if (temp > diametersThreads[offset + points[i].belongTo])
					diametersThreads[offset + points[i].belongTo] = temp;
			}
		}
	}
	for (i = 0; i < numOfClusters; i++)
	{
		diameters[i] = diametersThreads[i];
		for (j = 1; j < maxThreads; j++)
		{
			if (diameters[i] < diametersThreads[j * numOfClusters + i])
				diameters[i] = diametersThreads[j * numOfClusters + i];
		}
	}
	return diameters;
}

void calculateCenters(point_t* points, int numOfPoints, cluster_t* clusters, int numOfClusters)
{
	calculateSumCenters(points, numOfPoints, clusters, numOfClusters);
	synchronizedCenters(clusters, numOfClusters);
}

void dividePointToClusters(cluster_t* clusters, double* clustersCoords, int numOfClusters, point_t* points, double* pointsCoordsCUDAPointer, int numOfPoints, int dimension,int* change)
{
	cudaError_t status;
	int i, procID,*belongTo = NULL;
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);
	*change = 0;
	status = dividePointToClustersWithCuda(pointsCoordsCUDAPointer, numOfPoints, clustersCoords, numOfClusters, dimension, &belongTo);
	if (status != cudaSuccess)
		printf("cuda Failed!");
	for (i = 0; i < numOfClusters; i++)
		clusters[i].numOfPointsInProc = 0;
	for (i = 0; i < numOfPoints; i++)
	{
		if (points[i].belongTo != belongTo[i])
			*change = 1;
		points[i].belongTo = belongTo[i];
		clusters[belongTo[i]].numOfPointsInProc++;
	}
	free(belongTo);
}

void dividePointToClustersWithOmp(cluster_t* clusters, double* clustersCoords, int numOfClusters, point_t* points, double* pointsCoords, int numOfPoints, int dimension,int* change)
{
	int i, j, minCluster = 0, changeTemp = 0;
	double minDistance, distanceTemp;

	*change = 0;
	for (i = 0; i < numOfClusters; i++)
		clusters[i].numOfPointsInProc = 0;

#pragma omp parallel for private(j,minCluster,minDistance,distanceTemp) reduction(| :changeTemp)
	for (i = 0; i < numOfPoints; i++)
	{
		minCluster = 0;
		minDistance = distance((clusters[0].center), (points[i].coordinations), dimension);
		for (j = 1; j < numOfClusters; j++)
		{
			distanceTemp = distance((clusters[j].center), (points[i].coordinations), dimension);
			if (distanceTemp < minDistance)
			{
				minDistance = distanceTemp;
				minCluster = j;
			}
		}
		if (points[i].belongTo != minCluster)
			changeTemp = 1;
		points[i].belongTo = minCluster;
	}
	if (changeTemp != 0)
		*change = 1;
	else
		*change = 0;
	for (i = 0; i < numOfPoints; i++)
		clusters[points[i].belongTo].numOfPointsInProc++;
}