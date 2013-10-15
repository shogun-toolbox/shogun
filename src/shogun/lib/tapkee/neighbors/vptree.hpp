/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Laurens van der Maaten, Sergey Lisitsyn
 */

#ifndef TAPKEE_VPTREE_H_
#define TAPKEE_VPTREE_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/defines.hpp>
/* End of Tapkee includes */

#include <vector>
#include <queue>
#include <algorithm>
#include <limits>

namespace tapkee
{
namespace tapkee_internal
{

template<class Type, class RandomAccessIterator, class DistanceCallback>
struct compare_impl;

template<class RandomAccessIterator, class DistanceCallback>
struct DistanceComparator
{
	DistanceCallback callback;
	const RandomAccessIterator item;
	DistanceComparator(const DistanceCallback& c, const RandomAccessIterator& i) :
		callback(c), item(i) {}
	inline bool operator()(const RandomAccessIterator& a, const RandomAccessIterator& b)
	{
		return compare_impl<typename DistanceCallback::type,RandomAccessIterator,DistanceCallback>()
			(callback,item,a,b);
	}
};

struct KernelType;

template<class RandomAccessIterator, class DistanceCallback>
struct compare_impl<KernelType,RandomAccessIterator,DistanceCallback>
{
	inline bool operator()(DistanceCallback& callback, const RandomAccessIterator& item,
	                       const RandomAccessIterator& a, const RandomAccessIterator& b)
	{
		return (-2*callback(item,a) + callback(a,a)) < (-2*callback(item,b) + callback(b,b));
	}
};

struct DistanceType;

template<class RandomAccessIterator, class DistanceCallback>
struct compare_impl<DistanceType,RandomAccessIterator,DistanceCallback>
{
	inline bool operator()(DistanceCallback& callback, const RandomAccessIterator& item,
	                       const RandomAccessIterator& a, const RandomAccessIterator& b)
	{
		return callback(item,a) < callback(item,b);
	}
};

template<class RandomAccessIterator, class DistanceCallback>
class VantagePointTree
{
public:

	// Default constructor
	VantagePointTree(RandomAccessIterator b, RandomAccessIterator e, DistanceCallback c) :
		begin(b), items(), callback(c), tau(0.0), root(0)
	{
		items.reserve(e-b);
		for (RandomAccessIterator i=b; i!=e; ++i)
			items.push_back(i);
		root = buildFromPoints(0, items.size());
	}

	// Destructor
	~VantagePointTree()
	{
		delete root;
	}

	// Function that uses the tree to find the k nearest neighbors of target
	std::vector<IndexType> search(const RandomAccessIterator& target, int k)
	{
		std::vector<IndexType> results;
		// Use a priority queue to store intermediate results on
		std::priority_queue<HeapItem> heap;

		// Variable that tracks the distance to the farthest point in our results
		tau = std::numeric_limits<double>::max();

		// Perform the searcg
		search(root, target, k, heap);

		// Gather final results
		results.reserve(k);
		while(!heap.empty()) {
			results.push_back(items[heap.top().index]-begin);
			heap.pop();
		}
		return results;
	}

private:

	VantagePointTree(const VantagePointTree&);
	VantagePointTree& operator=(const VantagePointTree&);

	RandomAccessIterator begin;
	std::vector<RandomAccessIterator> items;
	DistanceCallback callback;
	double tau;

	struct Node
	{
		int index;
		double threshold;
		Node* left;
		Node* right;

		Node() :
			index(0), threshold(0.),
			left(0), right(0)
		{
		}

		~Node()
		{
			delete left;
			delete right;
		}

		Node(const Node&);
		Node& operator=(const Node&);

	}* root;

	struct HeapItem {
		HeapItem(int i, double d) :
			index(i), distance(d) {}
		int index;
		double distance;
		bool operator<(const HeapItem& o) const {
			return distance < o.distance;
		}
	};


	Node* buildFromPoints(int lower, int upper)
	{
		if (upper == lower)
		{
			return NULL;
		}

		Node* node = new Node();
		node->index = lower;

		if (upper - lower > 1)
		{
			int i = (int) (tapkee::uniform_random() * (upper - lower - 1)) + lower;
			std::swap(items[lower], items[i]);

			int median = (upper + lower) / 2;
			std::nth_element(items.begin() + lower + 1, items.begin() + median, items.begin() + upper,
				DistanceComparator<RandomAccessIterator,DistanceCallback>(callback,items[lower]));

			node->threshold = callback.distance(items[lower], items[median]);
			node->index = lower;
			node->left = buildFromPoints(lower + 1, median);
			node->right = buildFromPoints(median, upper);
		}

		return node;
	}

	void search(Node* node, const RandomAccessIterator& target, int k, std::priority_queue<HeapItem>& heap)
	{
		if (node == NULL)
			return;

		double distance = callback.distance(items[node->index], target);

		if (distance < tau)
		{
			if (heap.size() == static_cast<size_t>(k))
				heap.pop();

			heap.push(HeapItem(node->index, distance));

			if (heap.size() == static_cast<size_t>(k))
				tau = heap.top().distance;
		}

		if (node->left == NULL && node->right == NULL)
		{
			return;
		}

		if (distance < node->threshold)
		{
			if ((distance - tau) <= node->threshold)
				search(node->left, target, k, heap);

			if ((distance + tau) >= node->threshold)
				search(node->right, target, k, heap);
		}
		else
		{
			if ((distance + tau) >= node->threshold)
				search(node->right, target, k, heap);

			if ((distance - tau) <= node->threshold)
				search(node->left, target, k, heap);
		}
	}
};

}
}
#endif
