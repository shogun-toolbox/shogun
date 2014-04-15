/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Zharmagambetov Arman, kd tree data structure
 */

#ifndef _KDTREE_H_
#define _KDTREE_H_

#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <cmath>
#include <vector>
#include <cassert>
#include <limits>
#include <algorithm>

using namespace std;
using namespace shogun;

/*
 * The class, which represents one node of the kd-tree
 */
class KDTNode {
	friend class CKDTree;

protected:
	KDTNode *left, *right;	// left/right child
	vector<float> data;		// the actual data item
	// access to the node contents
	// this is a bit dangerous, hence private
	inline vector<float>& content() {
		return data;
	}

public:
	KDTNode(const vector<float>& x) : left(NULL), right(NULL), data(x) {}

	inline KDTNode* get_left() {
		return left;
	}
	inline KDTNode* get_right() {
		return right;
	}
	inline vector<float> get_data() {
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


/*
 * Kd-tree data structure, it raises the applying of KNN to O(logn),
 * but requires time to train
 */
class CKDTree
{
	/** Kd-tree implementation area */
	protected:
		KDTNode *root;	// root of tree
		int count;        // size of tree
		int dimensionOfFeatures;

	public:
		CKDTree() : root(NULL), count(0) {}

		inline int size() const {
			return count;
		}

		//Constructs a kd-tree from the specified collection of elements.
		void train_kd(CDenseFeatures<float64_t> elements, SGVector<int32_t> labels);

		//Find ONE nearest neighbor
		vector<float> FindNearestNeighbor(vector<float> location);
		
		// Finds the N values in the tree that are nearest to the specified location.
		vector<vector<float> > FindNearestNNeighbor(vector<float> location, int numNeighbors);

		// basic classification algorithm, it finds nearest n neighbors and determine the
		// class of input vector
		float ClassifyKNierestNeighbor(vector<float> location, int k_nn);
		
		// transform SGMatrix to 2d vetor
		vector<vector<float> > featureToVector(SGMatrix<float64_t> features);

	protected:
		
		bool vectorComparer(vector<float> v1, vector<float> v2);

		// Recursively construct kd-tree from input array of features
		KDTNode* construct_helper(KDTNode*& nodep, int dimension, int startIndex, int endIndex,
									int depth, vector<vector<float> > elements);
		
		// sort features in particular dimension, i.e. it implements quicksort algorithm
		void sortByDimension(vector<vector<float> >& elements, int start, int end, int currentDim);

		// helper for quicksort algorithm, it returns middle element, and
		// reorganize elements
		int sortPartition(vector<vector<float> >& elements, int start, int end, int currentDim);
		
		// Calculate distance by Euclidean
		double distanceByEuclidean(vector<float> v1, vector<float> v2);
		
		// Recursively iterate through leafs of kd-tree to find the nearest neighbour
		vector<float> FindNearestNeighborHelper(vector<float> location, KDTNode*& nodep,
			vector<float> bestValue, double bestDistance, int depth);
		
		// reqursively search in the tree for nearest n neighbors and insert it in priority queue
		void FindNearestNNeighborHelper(vector<float> location, KDTNode*& nodep, vector<float>& bestValue, double& bestDistance,
			int numNeighbors, vector<vector<float> >& valuesList, int depth);

		// sort features matrix and reorganise it as priority queue , i.e. it implements quicksort algorithm
		void sortByDistance(vector<vector<float> >& valuesList, int start, int end, vector<float> location);
	
		// helper for quicksort algorithm, it returns middle element, and
 		// reorganize elements
		int sortDistancePartition(vector<vector<float> >& valuesList, int start, int end, vector<float> location);
};

/** KNN classifier with KD-tree*/

#endif
