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

// FibonacciHeap.cpp: implementation of the CMFibonacciHeap class.
//
//////////////////////////////////////////////////////////////////////

#include "lib/FibonacciHeap.h"

using namespace shogun;

CFibonacciHeap::CFibonacciHeap(void)
{
	max_num_trees = 0;
	max_num_nodes = 0;

	trees = NULL;
	nodes = NULL;

	num_items = 0;
	sum_tree = 0;
}

CFibonacciHeap::CFibonacciHeap(int32_t capacity)
{
    int32_t i;

    max_num_trees = 1 + (int32_t)(1.44 * CMath::log2((float32_t)capacity));
    max_num_nodes = capacity;

    trees = new FibonacciHeapNode* [max_num_trees];
    for(i = 0; i < max_num_trees; i++) trees[i] = NULL;

    nodes = new FibonacciHeapNode* [capacity];
    for(i = 0; i < capacity; i++) nodes[i] = NULL;

    num_items = 0;

    /* The sum_tree of the heap helps to keep track of the maximum rank while
     * nodes are inserted or deleted.
     */
    sum_tree = 0;
}

CFibonacciHeap::~CFibonacciHeap(void)
{
	for(int32_t i = 0; i < max_num_nodes; i++)
		delete nodes[i];

	delete[] nodes;
	delete[] trees;
}

void CFibonacciHeap::insert(int32_t item, float64_t key)
{
	FibonacciHeapNode* new_node;

	new_node = new FibonacciHeapNode;
	new_node->child = NULL;

	new_node->left = new_node;
	new_node->right = new_node;

	new_node->rank = 0;
	new_node->item = item;
	new_node->key = key;

	nodes[item] = new_node;
	meld(new_node);
	num_items++;
}

int32_t CFibonacciHeap::delete_min(void)
{
	FibonacciHeapNode *min_node, *child, *next;
	float64_t key, key_a;
	int32_t temp_rank, temp_sum, temp_item;

	/* First we determine the maximum rank in the heap. */
	temp_sum = sum_tree;
	temp_rank = -1;

	while(temp_sum)
	{
		temp_sum >>= 1;
		temp_rank++;
	}

	/* Now determine which root node is the minimum. */
	min_node = trees[temp_rank];
	key = min_node->key;

	while(temp_rank > 0)
	{
		temp_rank--;
		next = trees[temp_rank];

		if(next)
		{
			key_a = next->key;
			if(key_a < key)
			{
				key = key_a;
				min_node = next;
			}
		}
	}

	/* We remove the minimum node from the heap but keep a pointer to it. */
	temp_rank = min_node->rank;
	trees[temp_rank] = NULL;

	sum_tree -= (1 << temp_rank);

	child = min_node->child;
	if(child)
		meld(child);

	/* Record the vertex no of the old minimum node before deleting it. */
	temp_item = min_node->item;
	nodes[temp_item] = NULL;

	delete min_node;

	num_items--;

	return temp_item;
}

void CFibonacciHeap::decrease_key(int32_t item, float64_t key)
{
	FibonacciHeapNode *cut_node, *new_roots;
	FibonacciHeapNode *parent, *left, *right;

	int32_t prev_rank;

	/* Obtain a pointer to the decreased node and its parent then decrease the
	 * nodes key.
	 */
	cut_node = nodes[item];
	parent = cut_node->parent;
	cut_node->key = key;

	/* No reinsertion occurs if the node changed was a root. */
	if(!parent)
		return;

	/* Update the left and right pointers of cutNode and its two neighbouring
	 * nodes.
	 */
	left = cut_node->left;
	right = cut_node->right;
	left->right = right;
	right->left = left;

	cut_node->left = cut_node;
	cut_node->right = cut_node;

	/* Initially the list of new roots contains only one node. */
	new_roots = cut_node;

	/* While there is a parent node that is marked a cascading cut occurs. */
	while(parent && parent->marked)
	{
		/* Decrease the rank of cutNode's parent and update its child pointer.
		 */
		parent->rank--;
		if(parent->rank)
		{
			if(parent->child == cut_node)
				parent->child = right;
		}
		else
		{
			parent->child = NULL;
		}

		/* Update the cutNode and parent pointers to the parent. */
		cut_node = parent;
		parent = cut_node->parent;

		/* Update the left and right pointers of cutNodes two neighbouring
		 * nodes.
		 */
		left = cut_node->left;
		right = cut_node->right;

		left->right = right;
		right->left = left;

		/* Add cutNode to the list of nodes to be reinserted as new roots. */
		left = new_roots->left;

		new_roots->left = cut_node;
		left->right = cut_node;

		cut_node->left = left;
		cut_node->right = new_roots;

		new_roots = cut_node;
	}

	/* If the root node is being relocated then update the trees[] array.
	 * Otherwise mark the parent of the last node cut.
	 */

	if(!parent)
	{
		prev_rank = cut_node->rank + 1;
		trees[prev_rank] = NULL;
		sum_tree -= (1 << prev_rank);
	}
	else
	{
		/* Decrease the rank of cutNode's parent an update its child pointer.
		 */
		parent->rank--;
		if(parent->rank)
		{
			if(parent->child == cut_node)
				parent->child = right;
		}
		else
		{
			parent->child = NULL;
		}

		parent->marked = true;
	}

	/* Meld the new roots into the heap. */
	meld(new_roots);
}

void CFibonacciHeap::meld(FibonacciHeapNode* tree_list)
{
	FibonacciHeapNode *first, *next, *node_ptr, *new_root;
	FibonacciHeapNode *temp, *temp_a, *left, *right;

	int32_t temp_rank;

	/* We meld each tree in the circularly linked list back into the root level
	 * of the heap.  Each node in the linked list is the root node of a tree.
	 * The circularly linked list uses the sibling pointers of nodes.  This
	 *  makes melding of the child nodes from a deleteMin operation simple.
	 */

	node_ptr = tree_list;
	first = tree_list;

	do
	{
		/* Keep a pointer to the next node and remove sibling and parent links
		 * from the current node.  nodePtr points to the current node.
		 */

		next = node_ptr->right;

		node_ptr->right = node_ptr;
		node_ptr->left = node_ptr;

		/* We merge the current node, nodePtr, by inserting it into the
		 * root level of the heap.
		 */

		new_root = node_ptr;
		temp_rank = node_ptr->rank;

		/* This loop inserts the new root into the heap, possibly restructuring
		 * the heap to ensure that only one tree for each degree exists.
		 */

		do
		{
			/* Check if there is already a tree of degree r in the heap.
			 * If there is then we need to link it with newRoot so it will be
			 * reinserted into a new place in the heap.
			 */

			temp = trees[temp_rank];
			if(temp)
			{
				/* temp will be linked to newRoot and relocated so we no
				 * longer will have a tree of degree r.
				 */

				trees[temp_rank] = NULL;
				sum_tree -= (1 << temp_rank);

				/* Swap temp and newRoot if necessary so that newRoot always
				 * points to the root node which has the smaller key of the
				 * two.
				 */

				if(temp->key < new_root->key)
				{
					temp_a = new_root;
					new_root = temp;
					temp = temp_a;
				}

				/* Link temp with newRoot, making sure that sibling pointers
				 * get updated if rank is greater than 0.  Also, increase r for
				 * the next pass through the loop since the rank of new has
				 * increased.
				 */

				if(temp_rank++ > 0)
				{
					right = new_root->child;
					left = right->left;

					temp->left = left;
					temp->right = right;

					left->right = temp;
					right->left = temp;
				}

				new_root->child = temp;
				new_root->rank = temp_rank;

				temp->parent = new_root;
				temp->marked = false;
			}
			else
			{
				/* Otherwise if there is not a tree of degree r in the heap we
				 * allow newRoot, which possibly carries moved trees in the heap,
				 * to be a tree of degree r in the heap.
				 */

				trees[temp_rank] = new_root;
				sum_tree += (1 << temp_rank);

				/* NOTE:  Because newRoot is now a root we ensure it is
				 *        marked.
				 */
				new_root->marked = true;
			}

		} while(temp);

		node_ptr = next;

	} while(node_ptr != first);
}



