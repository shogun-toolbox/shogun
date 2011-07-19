/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 20?? Shane Saunders (University of Canterbury)
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef FIBONACCI_H_
#define FIBONACCI_H_

#include "base/SGObject.h"
#include "lib/common.h"
#include "lib/Mathematics.h"

namespace shogun
{

struct FibonacciHeapNode
{
	/** pointer to parent node */
	FibonacciHeapNode* parent;
	/** pointer to child node */
	FibonacciHeapNode* child;
	/** pointer to left sibling */
	FibonacciHeapNode* left;
	/** pointer to right sibling */
	FibonacciHeapNode* right;
	/** rank of node */
	int32_t rank;
	/** marked flag */
	bool marked;
	/** key of node */
	float64_t key;
	/** item of node */
	int32_t item;
};

/** @brief the class FibonacciHeap, a fibonacci
 * heap. Generally used by Isomap for Dijkstra heap
 * algorithm
 */
class CFibonacciHeap: public CSGObject
{
public:
	/** empty constructor */
	CFibonacciHeap();

	/** constructor for heap with specified capacity */
	CFibonacciHeap(int32_t capacity);

	virtual inline const char* get_name() const
	{
		return "FibonacciHeap";
	}

	/** destructor */
	virtual ~CFibonacciHeap();

	/**
	 *
	 */
	int32_t get_num_items() const
	{
		return num_items;
	}

	/**
	 *
	 */
	void insert(int32_t item, float64_t key);

	/** deletes and returns item with minimal key.
	 * Have amortized time of O(log n)
	 * @return item with minimal key
	 */
	int32_t delete_min();

	/**
	 *
	 */
	void decrease_key(int32_t item, float64_t key);

protected:

	/** trees */
	FibonacciHeapNode** trees;

	/** nodes */
	FibonacciHeapNode** nodes;

	/** maximal number of nodes */
	int32_t max_num_nodes;

	/** maximal number of trees */
	int32_t max_num_trees;

	/** current number of items */
	int32_t num_items;

	/** wtf */
	int32_t sum_tree;

private:
	void meld(FibonacciHeapNode* tree_list);
};

}
#endif /* FIBONACCI_H_ */
