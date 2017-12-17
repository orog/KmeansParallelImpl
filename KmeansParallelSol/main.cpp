#define _CRT_SECURE_NO_WARNINGS
#include "KmeansParallel.h"
#include "MPI_Kmeans.h"
#include "cluster.h"
#include "kernel.h"
#define TRUE 1
#define FALSE 0
#define ROOT 0

int main(int argc, char *argv[])
{
	int myDatabaseSize,numOfProcs, procID,*belongTo = NULL;
	kmean_info_t info;
	cluster_t* clusters = NULL;
	point_t* pointsDatabase = NULL, *myPoints = NULL;
	double* coordsDatabase = NULL, *myPointsCoords = NULL,*myClusterCoords = NULL, *myPointsCordsCUDA;
	char* dataSetPath = "D:\\Sales_Transactions_Dataset_Weekly.txt";
	char* outputFile = "D:\\kmeans_results.txt";
	MPI_Datatype pointDatatype;
	MPI_Datatype clusterDatatype;
	MPI_Datatype infoDatatype;
	double startTime, finishedTime;

	//intialMPI
	intialMPI(&procID, &numOfProcs, &argc, &argv);
	createInfoDatatype(&infoDatatype);
	createClusterDatatype(&clusterDatatype);
	createPointDatatype(&pointDatatype);
	
	if (procID == ROOT)
	{
/*************************************      Phase 1 - Intialize database    ************************************************************************/
		read_file(dataSetPath, &info.numOfPoints, &info.maxNumOfClusters, &info.maxNumOfIter, &info.qualityMeasure,&info.dimension, &pointsDatabase,&coordsDatabase);
		info.currentQuality = info.qualityMeasure + 1;
		info.currentNumOfClusters = 2;

/*************************************      Phase 2 - Send data to the processes    ****************************************************************/
		startTime = MPI_Wtime();
		info.salveDatabaseSize = info.numOfPoints / numOfProcs;
		myDatabaseSize = info.salveDatabaseSize + info.numOfPoints % numOfProcs;
	}
	
	MPI_Bcast(&info, 1, infoDatatype, ROOT, MPI_COMM_WORLD);

	if(procID != ROOT)
		myDatabaseSize = info.salveDatabaseSize;

	//send and recive points database
	scattervDatabase(pointsDatabase, coordsDatabase, myDatabaseSize, info.salveDatabaseSize, ROOT, info.dimension, pointDatatype, &myPoints, &myPointsCoords);
	
	
	copyPointsCordToCUDA(myPointsCoords, myDatabaseSize, info.dimension, &myPointsCordsCUDA);
/*************************************      START K-means with quality measurement    **************************************************************/

	while (info.currentNumOfClusters <= info.maxNumOfClusters && info.currentQuality >= info.qualityMeasure)
	{
/*************************************      Phase 3 - Broadcasting Clusters to all process    *******************************************************/
        free(clusters);
        free(myClusterCoords);
		if (procID == ROOT)
			intialClusters(&clusters, &myClusterCoords, info.dimension, info.currentNumOfClusters, pointsDatabase);

		bCastCenters(&clusters, &myClusterCoords, info.currentNumOfClusters, info.dimension, clusterDatatype, ROOT);
        
/*************************************      Phase 4 - K-means    ************************************************************************************/
		
		ParallelKmeansAlg(clusters, myClusterCoords, info.currentNumOfClusters, myPoints, myPointsCoords,myPointsCordsCUDA, myDatabaseSize, info.dimension , info.maxNumOfIter);

/*************************************      Phase 5 - Calculate Clusters Quality   *****************************************************************/

		gathervBelongTo(pointsDatabase, info.numOfPoints, myPoints, myDatabaseSize, info.salveDatabaseSize, pointDatatype, ROOT);
		
		if (procID == ROOT)
		{
			info.currentQuality = calcQuality(clusters,info.currentNumOfClusters, pointsDatabase,info.numOfPoints,info.dimension);
		}

		MPI_Bcast(&info.currentQuality, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

/*************************************      Phase 6 - Check Clusters Quality    ********************************************************************/
		if (info.currentQuality > info.qualityMeasure)
		{	
			
			info.currentNumOfClusters++;
		}

/*************************************      THE END    *********************************************************************************************/
	}
	
	if (procID == ROOT)
	{
        if(info.currentNumOfClusters > info.maxNumOfClusters)
            info.currentNumOfClusters--;
		finishedTime = MPI_Wtime();
		printf("Time : %lf", finishedTime - startTime);
		output_file(outputFile, clusters, info.currentNumOfClusters, info.currentQuality);
		free(pointsDatabase);
		free(coordsDatabase);
	}
	cudaFree(myPointsCordsCUDA);
	free(myPoints);
	free(myPointsCoords);
	free(clusters);
	free(myClusterCoords);
	MPI_Finalize();
}
