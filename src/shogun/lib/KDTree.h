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
#include <limits>
#include <algorithm>

using namespace std;
using namespace shogun;

/**
 * The class, which represents one node of the kd-tree
 */
class KDTNode {
	friend class CKDTree;

protected:
	KDTNode *left, *right;	// left/right child
	vector<float> data;		// the actual data item
	
	/** access to the node contents
	* this is a bit dangerous, hence private
	*/
	inline vector<float>& content() {
		return data;
	}

public:
	/** constructor
	* @param x represents on feature
	*/
	KDTNode(const vector<float>& x) : left(NULL), right(NULL), data(x) {}

	/** get left subtree
	* @return left node
	*/
	inline KDTNode* get_left() {
		return left;
	}
	
	/** get left subtree
	* @return left node
	*/
	inline KDTNode* get_right() {
		return right;
	}
	
	/** get left subtree
	* @return left node
	*/
	inline vector<float> get_data() {
		return data;
	}

	/** remove whole subtree
	* note that one must unlink a node
	* before destroying it in KDTree::remove()
	* otherwise the whole subtree will disappear
	*/
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
		/** constructor
		 *
		 * @param root the root of whole tree
		 * @param count the number of elements
		 */
		CKDTree() : root(NULL), count(0) {}
		
		/** the actual size of the tree
		* @return the size of the tree
		*/
		inline int size() const {
			return count;
		}
		
		/** Constructs a kd-tree from the specified collection of elements.
		*
		* @param elements training data (parameter can be avoided if distance or
		* kernel-based classifiers are used and distance/kernels are
		* initialized with train data)
		*
		* @param labels of trained data
		*/
		void train_kd(CDenseFeatures<float64_t>* elements, SGVector<int32_t> labels);
		
		/**Find ONE nearest neighbor
		* @param location one entry data which need to classify
		* @return the nearest point to the data
		*/
		vector<float> find_nearest_neighbor(vector<float> location);
		
		/** Finds the N values in the tree that are nearest to the specified location.
		* @param location one entry data which need to classify
		* @return nearest N points
		*/
		vector<vector<float> > find_nearest_nneighbor(vector<float> location, int numNeighbors);
		
		/** basic classification algorithm, it finds nearest n neighbors and determine the
		* class of input vector
		* @param location one entry data which need to classify
		* @return class which most related to entry data
		*/
		float classify_knierest_neighbor(vector<float> location, int k_nn);
		
		/** transform SGMatrix to 2d vetor
		* @param feature matrix
		* @return two dimensional array
		*/
		vector<vector<float> > feature_to_vector(SGMatrix<float64_t> features);

	protected:
	
		/** Recursively construct kd-tree from input array of features
		* @param nodep pointer to the current node of the tree
		* @param dimension dimension to differentiate the points
		* @param startIndex the first point
		* @param endIndex the last point in the tree
		* @param depth is used two sort points in a plane
		* @param elements the feature matrix
		*
		* @return the place of the point in our kd-tree
		*/
		KDTNode* construct_helper(KDTNode*& nodep, int dimension, int startIndex, int endIndex,
									int depth, vector<vector<float> > elements);
		
		/** sort features in particular dimension, i.e. it implements quicksort algorithm
		* @param matrix elements used to construct kd-tree
		* @param the first point
		* @param the last point in the tree
		* @param dimension used to sort the points
		*/
		void sort_by_dimension(vector<vector<float> >& elements, int start, int end, int currentDim);
		
		/** helper for quicksort algorithm, it returns middle element, and
		* reorganize elements
		* @param matrix elements used to construct kd-tree
		* @param the first point
		* @param the last point in the tree
		* @param dimension used to sort the points
		*
		* @return index of pivot element
		*/
		int sort_partition(vector<vector<float> >& elements, int start, int end, int currentDim);
		
		/** Calculate distance by Euclidean
		* @param v1 first feature(point coordinates)
		* @param v2 second feature(point coordinates)
		*
		* @return distance between points
		*/
		double distance_by_euclidean(vector<float> v1, vector<float> v2);
		
		/** Recursively iterate through leafs of kd-tree to find the nearest neighbour
		* @param location one entry data which need to classify
		* @param current node of the tree
		* @param best value, the nearest distance point
		* @param nearest distance
		* @param depth to which locate the node of the tree
		*
		* @return nearest point
		*/
		vector<float> find_nearest_neighbor_helper(vector<float> location, KDTNode*& nodep,
			vector<float> bestValue, double bestDistance, int depth);
		
		/** reqursively search in the tree for nearest n neighbors and insert it in priority queue
		* @param location one entry data which need to classify
		* @param current node of the tree
		* @param best value, the nearest distance point
		* @param nearest distance
		* @param nearest n points
		* @param nearest point coordinates
		* @param depth to which locate the node of the tree
		*/
		void find_nearest_nneighbor_helper(vector<float> location, KDTNode*& nodep, vector<float>& bestValue, double& bestDistance,
			int numNeighbors, vector<vector<float> >& valuesList, int depth);
		
		/** sort features matrix and reorganise it as priority queue , i.e. it implements quicksort algorithm
		* @param matrix elements used to construct kd-tree
		* @param the first point
		* @param the last point in the tree
		* @param distance to other point
		*/
		void sort_by_distance(vector<vector<float> >& valuesList, int start, int end, vector<float> location);
		
		/** helper for quicksort algorithm, it returns middle element, and
 		* reorganize elements
 		* @param matrix elements used to construct kd-tree
		* @param the first point
		* @param the last point in the tree
		* @param distance to other point
		*
		* @return index of pivot element
		*/
		int sort_distance_partition(vector<vector<float> >& valuesList, int start, int end, vector<float> location);
};

/** KNN classifier with KD-tree*/

#endif
