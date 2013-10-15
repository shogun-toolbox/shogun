/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2011-2013 Evgeniy Andreev
 */

#ifndef TAPKEE_FIBONACCI_H_
#define TAPKEE_FIBONACCI_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/defines.hpp>
/* End of Tapkee includes */

#include <cmath>

namespace tapkee
{
namespace tapkee_internal
{

struct fibonacci_heap_node
{
	fibonacci_heap_node() : parent(NULL), child(NULL), left(NULL), right(NULL),
		rank(0), marked(false), index(-1), key(0.0)
	{
	}

	/** pointer to parent node */
	fibonacci_heap_node* parent;

	/** pointer to child node */
	fibonacci_heap_node* child;

	/** pointer to left sibling */
	fibonacci_heap_node* left;

	/** pointer to right sibling */
	fibonacci_heap_node* right;

	/** rank of node */
	int rank;

	/** marked flag */
	bool marked;

	/** index in heap */
	int index;

	/** key of node */
	ScalarType key;

private:
	fibonacci_heap_node(const fibonacci_heap_node& fh);
	fibonacci_heap_node& operator=(const fibonacci_heap_node& fh);
};

/** @brief the class fibonacci_heap, a fibonacci
 * heap. Generally used by Isomap for Dijkstra heap
 * algorithm
 *
 * w: http://en.wikipedia.org/wiki/Fibonacci_heap
 */
class fibonacci_heap
{
public:

	/** Constructor for heap with specified capacity */
	fibonacci_heap(int capacity) :
		min_root(NULL), nodes(NULL), num_nodes(0),
		num_trees(0), max_num_nodes(capacity), A(NULL), Dn(0)
	{
		nodes = (fibonacci_heap_node**)malloc(sizeof(fibonacci_heap_node*)*max_num_nodes);
		for (int i = 0; i < max_num_nodes; i++)
			nodes[i] = new fibonacci_heap_node;

		Dn = 1 + (int)(log(ScalarType(max_num_nodes))/log(2.));
		A = (fibonacci_heap_node**)malloc(sizeof(fibonacci_heap_node*)*Dn);
		for (int i = 0; i < Dn; i++)
			A[i] = NULL;

		num_nodes = 0;
		num_trees = 0;
	}

	~fibonacci_heap()
	{
		for(int i = 0; i < max_num_nodes; i++)
		{
			if(nodes[i] != NULL)
				delete nodes[i];
		}
		free(nodes);
		free(A);
	}

	/** Inserts nodes with certain key in array of nodes with index
	 * Have time of O(1)
	 */
	void insert(int index, ScalarType key)
	{
		if(index >= static_cast<int>(max_num_nodes) || index < 0)
			return;

		if(nodes[index]->index != -1)
			return; // node is not empty

		// init "new" node in array
		nodes[index]->child = NULL;
		nodes[index]->parent = NULL;

		nodes[index]->rank = 0;
		nodes[index]->index = index;
		nodes[index]->key = key;
		nodes[index]->marked = false;

		add_to_roots(nodes[index]);
		num_nodes++;
	}

	bool empty() const
	{
		return num_nodes==0;
	}

	int get_num_nodes() const
	{
		return num_nodes;
	}

	int get_num_trees()
	{
		return num_trees;
	}

	int get_capacity()
	{
		return max_num_nodes;
	}

	/** Deletes and returns item with minimal key
	 * Have amortized time of O(log n)
	 * @return item with minimal key
	 */
	int extract_min(ScalarType& ret_key)
	{
		fibonacci_heap_node *min_node;
		fibonacci_heap_node *child, *next_child;

		int result;

		if(num_nodes == 0)
			return -1;

		min_node = min_root;
		if(min_node == NULL)
			return -1; // heap is empty now

		child = min_node->child;
		while(child != NULL && child->parent != NULL)
		{
			next_child = child->right;

			// delete current child from childs list
			child->left->right = child->right;
			child->right->left = child->left;

			add_to_roots(child);

			// next iteration
			child = next_child;
		}

		// delete minimun from root list
		min_node->left->right = min_node->right;
		min_node->right->left = min_node->left;

		num_trees--;

		if(min_node == min_node->right)
		{
			min_root = NULL; // remove last element
		}
		else
		{
			min_root = min_node->right;
			consolidate();
		}

		result = min_node->index;
		ret_key = min_node->key;
		clear_node(result);

		num_nodes--;

		return result;
	}

	/** Clears all nodes in heap */
	void clear()
	{
		min_root = NULL;

		// clear all nodes
		for(int i = 0; i < max_num_nodes; i++)
		{
			clear_node(i);
		}

		num_nodes = 0;
		num_trees = 0;
	}

	/** Returns key by index
	 * @return -1 if not valid
	 */
	int get_key(int index, ScalarType& ret_key)
	{
		if(index >= max_num_nodes || index < 0)
			return -1;
		if(nodes[index]->index == -1)
			return -1;

		int result = nodes[index]->index;
		ret_key = nodes[index]->key;

		return result;
	}

	/** Decreases key by index
	 * Have amortized time of O(1)
	 */
	void decrease_key(int index, ScalarType& key)
	{
		fibonacci_heap_node* parent;

		if(index >= max_num_nodes || index < 0)
			return;
		if(nodes[index]->index == -1)
			return; // node is empty
		if(key > nodes[index]->key)
			return;


		nodes[index]->key = key;

		parent = nodes[index]->parent;

		if(parent != NULL && nodes[index]->key < parent->key)
		{
			cut(nodes[index], parent);
			cascading_cut(parent);
		}

		if(nodes[index]->key < min_root->key)
			min_root = nodes[index];
	}

private:

	fibonacci_heap();
	fibonacci_heap(const fibonacci_heap& fh);
	fibonacci_heap& operator=(const fibonacci_heap& fh);

private:
	/** Adds node to roots list */
	void add_to_roots(fibonacci_heap_node *up_node)
	{
		if(min_root == NULL)
		{
			// if heap is empty, node becomes circular root list
			min_root = up_node;

			up_node->left = up_node;
			up_node->right = up_node;
		}
		else
		{
			// insert node to root list
			up_node->right = min_root->right;
			up_node->left = min_root;

			up_node->left->right = up_node;
			up_node->right->left = up_node;

			// nomination of new minimum node
			if(up_node->key < min_root->key)
			{
				min_root = up_node;
			}
		}

		up_node->parent = NULL;
		num_trees++;
	}

	/** Consolidates heap */
	void consolidate()
	{
		fibonacci_heap_node *x, *y, *w;
		int d;

		for(int i = 0; i < Dn; i++)
		{
			A[i] = NULL;
		}

		min_root->left->right = NULL;
		min_root->left = NULL;
		w = min_root;

		do
		{
			x = w;
			d = x->rank;
			w = w->right;

			while(A[d] != NULL)
			{
				y = A[d];

				if(y->key < x->key)
				{
					fibonacci_heap_node *temp;

					temp = y;
					y = x;
					x = temp;
				}

				link_nodes(y, x);

				A[d] = NULL;
				d++;
			}
			A[d] = x;
		}
		while(w != NULL);

		min_root = NULL;
		num_trees = 0;

		for(int i = 0; i < Dn; i++)
		{
			if(A[i] != NULL)
			{
				A[i]->marked = false;
				add_to_roots(A[i]);
			}
		}
	}

	/** Links right node to childs of left node */
	void link_nodes(fibonacci_heap_node *y, fibonacci_heap_node *x)
	{
		if(y->right != NULL)
			y->right->left = y->left;
		if(y->left != NULL)
			y->left->right = y->right;

		num_trees--;

		y->left = y;
		y->right = y;

		y->parent = x;

		if(x->child == NULL)
		{
			x->child = y;
		}
		else
		{
			y->left = x->child;
			y->right = x->child->right;

			x->child->right = y;
			y->right->left = y;
		}

		x->rank++;

		y->marked = false;
	}

	/** Clears node by index */
	void clear_node(int index)
	{
		nodes[index]->parent = NULL;
		nodes[index]->child = NULL;
		nodes[index]->left = NULL;
		nodes[index]->right = NULL;

		nodes[index]->rank = 0;
		nodes[index]->index = -1;
		nodes[index]->key = 0;

		nodes[index]->marked = false;
	}

	/** Cuts child node from childs list of parent */
	void cut(fibonacci_heap_node *child, fibonacci_heap_node *parent)
	{
		if(parent->child == child)
			parent->child = child->right;

		if(parent->child == child)
			parent->child = NULL;

		parent->rank--;

		child->left->right = child->right;
		child->right->left = child->left;
		child->marked = false;

		add_to_roots(child);
	}

	void cascading_cut(fibonacci_heap_node* tree)
	{
		fibonacci_heap_node *temp;

		temp = tree->parent;
		if(temp != NULL)
		{
			if(!tree->marked)
			{
				tree->marked = true;
			}
			else
			{
				cut(tree, temp);
				cascading_cut(temp);
			}
		}
	}

protected:
	/** minimal root in heap */
	fibonacci_heap_node* min_root;

	/** array of nodes for fast search by index */
	fibonacci_heap_node** nodes;

	/** number of nodes */
	int num_nodes;

	/** number of trees */
	int num_trees;

	/** maximum number of nodes */
	int max_num_nodes;

	/** supporting array */
	fibonacci_heap_node **A;

	/** size of supporting array */
	int Dn;
};

}
}

#endif /* FIBONACCI_H_ */
