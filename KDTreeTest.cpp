#include <iostream>
#include <vector>
#include "KDTree.h"
using namespace std;

int main()
{
	KDTree<float> kdtree;

	vector<float> testSet1;
	testSet1.push_back(1);
	testSet1.push_back(2);

	vector<float> testSet2;
	testSet2.push_back(2);
	testSet2.push_back(1);

	vector<float> testSet3;
	testSet3.push_back(4);
	testSet3.push_back(1);

	vector<float> testSet4;
	testSet4.push_back(3);
	testSet4.push_back(3);

	vector<float> testSet5;
	testSet5.push_back(1);
	testSet5.push_back(4);

	vector<vector<float>> testSets;
	testSets.push_back(testSet1);
	testSets.push_back(testSet2);
	testSets.push_back(testSet3);
	testSets.push_back(testSet4);
	testSets.push_back(testSet5);

	//construct
	kdtree.Construct(testSets);


	vector<float> nntest;
	nntest.push_back(5);
	nntest.push_back(2);

	//find 1 nn
	vector<float> nn = kdtree.FindNearestNeighbor(nntest);

	//find 3 nn
	vector<vector<float>> testKNN = kdtree.FindNearestNNeighbor(nntest, 3);
	
	int b;

	system("pause");
	return 0;
}