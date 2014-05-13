#include <shogun/lib/KDTree.h>

void CKDTree::train_kd(CDenseFeatures<float64_t>* element, SGVector<int32_t> labels)
{
	SGMatrix<float64_t> features = element->get_feature_matrix();
	dimensionOfFeatures = features.num_rows;
	vector<vector<float> > elements = featureToVector(features);
	for (int i = 0; i < (int)elements.size(); i++)
	{
		assert ((int)elements[i].size() == dimensionOfFeatures);
		elements[i].push_back(labels[i]);
	}

	root = construct_helper(root, dimensionOfFeatures, 0, elements.size() - 1, 0, elements);
}

vector<vector<float> > CKDTree::featureToVector(SGMatrix<float64_t> features)
{
	vector<vector<float> > elem;
	for(int i = 0; i < features.num_cols; i++)
	{
		float64_t* cols = features.get_column_vector(i);
		vector<float> column(cols, cols + features.num_rows);
		elem.push_back(column);
	}
	return elem;
}

KDTNode* CKDTree::construct_helper(KDTNode*& nodep, int dimension, int startIndex,
	int endIndex, int depth, vector<vector<float> > elements)
{
	int length = endIndex - startIndex + 1;
	if (length == 0)
		return NULL;

	// Sort array of elements by component of chosen dimension, in ascending magnitude.
	int currentDim = depth % dimension;
	sortByDimension(elements, startIndex, endIndex, currentDim);

	// Select median element as pivot.
	int medianIndex = startIndex + length / 2;
	vector<float> medianElement = elements[medianIndex];

	// Create node and construct sub-trees around pivot element.
	nodep = new KDTNode(medianElement);
	nodep->left = construct_helper(nodep->left, dimension, startIndex, medianIndex - 1, depth + 1, elements);
	nodep->right = construct_helper(nodep->right, dimension, medianIndex + 1, endIndex, depth + 1, elements);

	return nodep;
}

void CKDTree::sortByDimension(vector<vector<float> >& elements, int start, int end, int currentDim)
{
	int middle;
	if (start < end)
	{
		middle = sortPartition(elements, start, end, currentDim);
		sortByDimension(elements, start, middle, currentDim);   // sort first section
		sortByDimension(elements, middle + 1, end, currentDim);    // sort second section
	}
	return;
}

int CKDTree::sortPartition(vector<vector<float> >& elements, int start, int end, int currentDim)
{
	vector<float> x = elements[start];
	int i = start - 1;
	int j = end + 1;
	vector<float> temp;
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

vector<float> CKDTree::FindNearestNeighbor(vector<float> location)
{
	assert(!location.empty());

	float tempClassifier = 0;
	location.push_back(tempClassifier);
	double maxDistance = numeric_limits<double>::max();
	return FindNearestNeighborHelper(location, root, root->data, maxDistance, 0);
}

vector<float> CKDTree::FindNearestNeighborHelper(vector<float> location, KDTNode*& nodep, vector<float> bestValue, double bestDistance, int depth)
{
	if (nodep == NULL)
		return bestValue;

	int curDimension = depth%dimensionOfFeatures;
	vector<float> nodeLocation = nodep->data;
	double distance = distanceByEuclidean(nodeLocation, location);

	// Check if current node is better than best node.
	// Current node cannot be same as search location.
	if (distance != 0 && distance < bestDistance)
	{
		bestValue = nodeLocation;
		bestDistance = distance;
	}

	// Check for best node in sub-tree of near child.
	KDTNode *nearChildNode = location[curDimension] - nodeLocation[curDimension] < 0 ?
		nodep->left : nodep->right;
	if (nearChildNode != NULL)
	{
		vector<float> nearBestValue = FindNearestNeighborHelper(location, nearChildNode, bestValue, bestDistance, depth + 1);
		double nearBestDistance = distanceByEuclidean(nearBestValue, location);
		bestValue = nearBestValue;
		bestDistance = nearBestDistance;
	}

	// Check whether splitting hyperplane given by current node intersects with hypersphere of current smallest
	// distance around given location.
	if (bestDistance > abs(location[curDimension] - nodeLocation[curDimension]))
	{
		// Check for best node in sub-tree of far child.
		KDTNode *farChildNode = nearChildNode == nodep->left ? nodep->right : nodep->left;
		if (farChildNode != NULL)
		{
			vector<float> farBestValue = FindNearestNeighborHelper(location, farChildNode, bestValue, bestDistance, depth + 1);
			double farBestDistance = distanceByEuclidean(farBestValue, location);
			bestValue = farBestValue;
			bestDistance = farBestDistance;
		}
	}

	return bestValue;
}

double CKDTree::distanceByEuclidean(vector<float> v1, vector<float> v2)
{
	double distance = 0.0;
	double sum = 0.0;

	assert (v1.size() == v2.size());

	//the last member is label, therefore iterate until last element
	for (size_t i = 0; i < v1.size() - 1; i++)
	{
		sum = sum + pow(abs(v1[i] - v2[i]), 2);
	}

	distance = sqrt(sum);

	return distance;
}

vector<vector<float> > CKDTree::FindNearestNNeighbor(vector<float> location, int numNeighbors)
{
	assert(!location.empty());

	float tempClassifier = 0;
	location.push_back(tempClassifier);
	vector<vector<float> > nodesList;
	vector<float> minBestValue = root->data;
	double minBestDistance = numeric_limits<double>::max();
	FindNearestNNeighborHelper(location,
		root,
		minBestValue,
		minBestDistance,
		numNeighbors,
		nodesList,
		0);

	return nodesList;
}


void CKDTree::FindNearestNNeighborHelper(vector<float> location, KDTNode*& nodep, vector<float>& bestValue, double& bestDistance,
	int numNeighbors, vector<vector<float> >& valuesList, int depth)
{
	if (nodep == NULL)
		return;

	int curDimension = depth%dimensionOfFeatures;
	vector<float> nodeLocation = nodep->data;
	double distance = distanceByEuclidean(nodeLocation, location);

	// Check if current node is better than best node.
	// Current node cannot be same as search location.
	if (distance != 0 && distance < bestDistance)
	{
		if ((int)valuesList.size() == numNeighbors)
			valuesList[valuesList.size() - 1] = nodeLocation;
		else valuesList.push_back(nodeLocation);

		sortByDistance(valuesList, 0, valuesList.size() - 1, location);

		if ((int)valuesList.size() == numNeighbors)
		{
			bestValue = valuesList[valuesList.size() - 1];
			bestDistance = distanceByEuclidean(location, bestValue);
		}
	}

	// Check for best node in sub-tree of near child.
	KDTNode *nearChildNode = location[curDimension] - nodeLocation[curDimension] < 0 ?
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
		KDTNode *farChildNode = nearChildNode == nodep->left ? nodep->right : nodep->left;
		if (farChildNode != NULL)
		{
			FindNearestNNeighborHelper(location, farChildNode, bestValue, bestDistance, numNeighbors,
				valuesList, depth + 1);
		}
	}

}

void CKDTree::sortByDistance(vector<vector<float> >& valuesList, int start, int end, vector<float> location)
{
	int middle;
	if (start < end)
	{
		middle = sortDistancePartition(valuesList, start, end, location);
		sortByDistance(valuesList, start, middle, location);   // sort first section
		sortByDistance(valuesList, middle + 1, end, location);    // sort second section
	}
	return;
}

int CKDTree::sortDistancePartition(vector<vector<float> >& valuesList, int start, int end, vector<float> location)
{
	vector<float> x = valuesList[start];
	int i = start - 1;
	int j = end + 1;
	vector<float> temp;
	do
	{
		do
		{
			j--;
		} while (distanceByEuclidean(x, location) < distanceByEuclidean(location, valuesList[j]));

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

float CKDTree::ClassifyKNierestNeighbor(vector<float> location, int k_nn)
{
	vector<vector<float> > knns = FindNearestNNeighbor(location, k_nn);
	vector<float> labelsOfNeighbors;

	if (knns.empty())
	{
		return 0;
	}

	int sizeOfFeatureVector = knns[0].size();

	for (size_t i = 0; i < knns.size(); i++)
	{
		labelsOfNeighbors.push_back(knns[i][sizeOfFeatureVector - 1]);
	}

	int max = 0;
	float mostFrequentLabel = labelsOfNeighbors[0];
	for (size_t i = 0; i < labelsOfNeighbors.size(); i++)
	{
		int countTemp = std::count(labelsOfNeighbors.begin(), labelsOfNeighbors.end(), labelsOfNeighbors[i]);
		if (countTemp > max){
			max = countTemp;
			mostFrequentLabel = labelsOfNeighbors[i];
		}
	}

	return mostFrequentLabel;
}

