// Represents a kd-tree data structure, used for the binary partitioning of k-dimensional space.
// This class implements simple KD tree structure without using Shogun or other extrenal libraries(from scratch).
// Features of this tree does not have labels.

#ifndef _KDTREE_H_
#define _KDTREE_H_

#include <cmath>
#include <vector>
#include <queue>

using namespace std;

// Forward references
template <typename T> class KDTree;
template <typename T> class KDTNode;

template <typename T>
class KDTNode {
	friend class KDTree<T>;

protected:
	KDTNode *left, *right;	// left/right child
	vector<T> data;		// the actual data item

	// access to the node contents
	// this is a bit dangerous, hence private
	vector<T>& content() {
		return data;
	}

public:
	KDTNode(const vector<T>& x) : left(NULL), right(NULL), data(x) {}

	KDTNode* get_left() {
		return left;
	}
	KDTNode* get_right() {
		return right;
	}
	vector<T> get_data() {
		return data;
	}

	// remove whole subtree
	// note that one must unlink a node
	// before destroying it in KDTree::remove()
	// otherwise the whole subtree will disappear
	~KDTNode() {
		if (left) {
			delete left;
		}
		if (right) {
			delete right;
		}
	}
};

/*!Begin Snippet:interface*/
template <typename T>
class KDTree {

protected:
	KDTNode<T> *root;	// root of tree
	int count;        // size of tree
	int dimensionOfFeatures;

public:
	KDTree() : root(NULL), count(0) {}

	virtual int size() const {
		return count;
	}

	void Construct(vector<vector<T>> elements);
	vector<T> FindNearestNeighbor(vector<T> location);
	vector<vector<T>> FindNearestNNeighbor(vector<T> location, int numNeighbors);
	
protected:
	bool vectorComparer(vector<T> v1, vector<T> v2);
	KDTNode<T>* construct_helper(KDTNode<T>*& nodep, int dimension, int startIndex, int endIndex, int depth, vector<vector<T>> elements);
	void sortByDimension(vector<vector<T>>& elements, int start, int end, int currentDim);
	int sortPartition(vector<vector<T>>& elements, int start, int end, int currentDim);
	double distanceByEuclidean(vector<T> v1, vector<T> v2);
	vector<T> FindNearestNeighborHelper(vector<T> location, KDTNode<T>*& nodep, vector<T> bestValue, double bestDistance, int depth);
	void FindNearestNNeighborHelper(vector<T> location, KDTNode<T>*& nodep, vector<T>& bestValue, double& bestDistance, 
		int numNeighbors, vector<vector<T>>& valuesList, int depth);
	void sortByDistance(vector<vector<T>>& valuesList, int start, int end, vector<T> location);
	int sortDistancePartition(vector<vector<T>>& valuesList, int start, int end, vector<T> location);
};

//Constructs a kd-tree from the specified collection of elements.
template<typename T>
void KDTree<T>::Construct(vector<vector<T>> elements)
{
	dimensionOfFeatures = elements[0].size();
	// Construct nodes of the tree.
	root = construct_helper(root, dimensionOfFeatures, 0, elements.size()-1, 0, elements);
}

//Recursively construct kd-tree from input array of features
template<typename T>
KDTNode<T>* KDTree<T>::construct_helper(KDTNode<T>*& nodep, int dimension, int startIndex, int endIndex, int depth, vector<vector<T>> elements)
{
	int length = endIndex - startIndex + 1;
	if (length == 0)
		return NULL;

	// Sort array of elements by component of chosen dimension, in ascending magnitude.
	int currentDim = depth % dimension;
	sortByDimension(elements, startIndex, endIndex, currentDim);

	// Select median element as pivot.
	int medianIndex = startIndex + length / 2;
	vector<T> medianElement = elements[medianIndex];

	// Create node and construct sub-trees around pivot element.
	nodep = new KDTNode<T>(medianElement);
	nodep->left = construct_helper(nodep->left, dimension, startIndex, medianIndex - 1, depth + 1, elements);
	nodep->right = construct_helper(nodep->right, dimension, medianIndex + 1, endIndex, depth + 1, elements);
	
	return nodep;
}

//sort features in particular dimension, i.e. it implements quicksort algorithm
template<typename T>
void KDTree<T>::sortByDimension(vector<vector<T>>& elements, int start, int end, int currentDim)
{
	// top = subscript of beginning of array
	// bottom = subscript of end of array

	int middle;
	if (start < end)
	{
		middle = sortPartition(elements, start, end, currentDim);
		sortByDimension(elements, start, middle, currentDim);   // sort first section
		sortByDimension(elements, middle + 1, end, currentDim);    // sort second section
	}
	return;
}

template<typename T>
int KDTree<T>::sortPartition(vector<vector<T>>& elements, int start, int end, int currentDim)
{
	vector<T> x = elements[start];
	int i = start - 1;
	int j = end + 1;
	vector<T> temp;
	do
	{
		do
		{
			j--;
		} while (x[currentDim] < elements[j][currentDim]);

		do
		{
			i++;
		} while (x[currentDim] > elements[i][currentDim]);

		if (i < j)
		{
			temp = elements[i];
			elements[i] = elements[j];
			elements[j] = temp;
		}
	} while (i < j);
	return j;           // returns middle subscript  
}

//Find ONE nearest neighbor 
template <typename T>
vector<T> KDTree<T>::FindNearestNeighbor(vector<T> location)
{
	if (location.empty())
		throw "Argument is null";

	return FindNearestNeighborHelper(location, root, root->data, distanceByEuclidean(root->data, location), 0);
}

//Recursively iterate through leafs of kd-tree to find the nearest neighbour
template <typename T>
vector<T> KDTree<T>::FindNearestNeighborHelper(vector<T> location, KDTNode<T>*& nodep, vector<T> bestValue, double bestDistance, int depth)
{
	if (nodep == NULL)
		return bestValue;

	int curDimension = depth%dimensionOfFeatures;
	vector<T> nodeLocation = nodep->data;
	double distance = distanceByEuclidean(nodeLocation, location);

	// Check if current node is better than best node.
	// Current node cannot be same as search location.
	if (distance != 0 && distance < bestDistance)
	{
		bestValue = nodeLocation;
		bestDistance = distance;
	}

	// Check for best node in sub-tree of near child.
	KDTNode<T> *nearChildNode = location[curDimension] - nodeLocation[curDimension] < 0 ?
		nodep->left : nodep->right;
	if (nearChildNode != NULL)
	{
		vector<T> nearBestValue = FindNearestNeighborHelper(location, nearChildNode, bestValue, bestDistance, depth + 1);
		double nearBestDistance = distanceByEuclidean(nearBestValue, location);
		bestValue = nearBestValue;
		bestDistance = nearBestDistance;
	}

	// Check whether splitting hyperplane given by current node intersects with hypersphere of current smallest
	// distance around given location.
	if (bestDistance > abs(location[curDimension] - nodeLocation[curDimension]))
	{
		// Check for best node in sub-tree of far child.
		KDTNode<T> *farChildNode = nearChildNode == nodep->left ? nodep->right : nodep->left;
		if (farChildNode != NULL)
		{
			vector<T> farBestValue = FindNearestNeighborHelper(location, farChildNode, bestValue, bestDistance, depth + 1);
			double farBestDistance = distanceByEuclidean(farBestValue, location);
			bestValue = farBestValue;
			bestDistance = farBestDistance;
		}
	}

	return bestValue;
}

//Calculate distance by Euclidean
template <typename T>
double KDTree<T>::distanceByEuclidean(vector<T> v1, vector<T> v2)
{
	double distance = 0.0;
	double sum = 0.0;

	if (v1.size() != v2.size())
		throw "Invalid arguments exception!";

	for (int i = 0; i < v1.size(); i++)
	{
		sum = sum + pow(abs(v1[i] - v2[i]), 2.0);
	}

	distance = sqrt(sum);

	return distance;
}

// Finds the N values in the tree that are nearest to the specified location.
template <typename T>
vector<vector<T>> KDTree<T>::FindNearestNNeighbor(vector<T> location, int numNeighbors)
{
	if (location.empty())
		throw "Argument is null";

	vector<vector<T>> nodesList;
	vector<T> minBestValue = root->data;
	double minBestDistance = 1000;

	FindNearestNNeighborHelper(location, 
		root, 
		minBestValue, 
		minBestDistance, 
		numNeighbors, 
		nodesList, 
		0);

	return nodesList;
}

template <typename T>
void KDTree<T>::FindNearestNNeighborHelper(vector<T> location, KDTNode<T>*& nodep, vector<T>& bestValue, double& bestDistance,
	int numNeighbors, vector<vector<T>>& valuesList, int depth)
{
	if (nodep == NULL)
		return;

	int curDimension = depth%dimensionOfFeatures;
	vector<T> nodeLocation = nodep->data;
	double distance = distanceByEuclidean(nodeLocation, location);

	// Check if current node is better than best node.
	// Current node cannot be same as search location.
	if (distance != 0 && distance < bestDistance)
	{
		if (valuesList.size() == numNeighbors)
			valuesList[valuesList.size() - 1] = nodeLocation;
		else valuesList.push_back(nodeLocation);
	    
		sortByDistance(valuesList, 0, valuesList.size() - 1, location);

		if (valuesList.size() == numNeighbors)
		{
			bestValue = valuesList[valuesList.size() - 1];
			bestDistance = distanceByEuclidean(location, bestValue);
		}
	}

	// Check for best node in sub-tree of near child.
	KDTNode<T> *nearChildNode = location[curDimension] - nodeLocation[curDimension] < 0 ?
		nodep->left : nodep->right;
	if (nearChildNode != NULL)
	{
		FindNearestNNeighborHelper(location, nearChildNode, bestValue, bestDistance, numNeighbors, 
			valuesList, depth + 1);
	}

	// Check whether splitting hyperplane given by current node intersects with hypersphere of current smallest
	// distance around given location.
	if (bestDistance > abs(location[curDimension] - nodeLocation[curDimension]))
	{
		// Check for best node in sub-tree of far child.
		KDTNode<T> *farChildNode = nearChildNode == nodep->left ? nodep->right : nodep->left;
		if (farChildNode != NULL)
		{
			FindNearestNNeighborHelper(location, farChildNode, bestValue, bestDistance, numNeighbors,
				valuesList, depth + 1);
		}
	}

}

template <typename T>
void KDTree<T>::sortByDistance(vector<vector<T>>& valuesList, int start, int end, vector<T> location)
{
	// top = subscript of beginning of array
	// bottom = subscript of end of array

	int middle;
	if (start < end)
	{
		middle = sortDistancePartition(valuesList, start, end, location);
		sortByDistance(valuesList, start, middle, location);   // sort first section
		sortByDistance(valuesList, middle + 1, end, location);    // sort second section
	}
	return;
}
template <typename T>
int KDTree<T>::sortDistancePartition(vector<vector<T>>& valuesList, int start, int end, vector<T> location)
{
	vector<T> x = valuesList[start];
	int i = start - 1;
	int j = end + 1;
	vector<T> temp;
	do
	{
		do
		{
			j--;
		} while (distanceByEuclidean(x, location) < distanceByEuclidean(location,valuesList[j]));

		do
		{
			i++;
		} while (distanceByEuclidean(x, location) > distanceByEuclidean(location, valuesList[i]));

		if (i < j)
		{
			temp = valuesList[i];
			valuesList[i] = valuesList[j];
			valuesList[j] = temp;
		}
	} while (i < j);
	return j;           // returns middle subscript  
}


#endif