#ifndef __KMEAN_INFO_H__
#define __KMEAN_INFO_H__

struct KmeanInfo
{
	int numOfPoints;
	int maxNumOfClusters;
	int maxNumOfIter;
	int currentNumOfClusters;
	int salveDatabaseSize;
	int dimension;
	double qualityMeasure;
	double currentQuality;
}typedef kmean_info_t;

#endif // !__KMEAN_INFO_H__

