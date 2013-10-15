/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2009-2013 John Langford, Dinoj Surendran, Fernando José Iglesias García
 */

#ifndef COVERTREE_H_
#define COVERTREE_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/neighbors/covertree_point.hpp>
/* End of Tapkee includes */

#include <cmath>
#include <limits>
#include <stdio.h>
#include <assert.h>

/* First written by John Langford jl@hunch.net
   Templatization by Dinoj Surendran dinojs@gmail.com
   Adaptation to Shogun by Fernando José Iglesias García
 */
namespace tapkee
{
namespace tapkee_internal
{

/**
 * Cover tree node TODO better doc
 */
template<class P>
struct node
{
	node() : p(), max_dist(0.0), parent_dist(0.0),
		children(NULL), num_children(0), scale(0)
	{
	}

	node(P _p, ScalarType _max_dist, ScalarType _parent_dist, node<P>* _children,
	     unsigned short int _num_children, short int _scale) : p(_p),
		max_dist(_max_dist), parent_dist(_parent_dist), children(_children),
		num_children(_num_children), scale(_scale)
	{
	}

	/** Point */
	P p;

	/** The maximum distance to any grandchild */
	ScalarType max_dist;

	/** The distance to the parent */
	ScalarType parent_dist;

	/** Pointer to the list of children of this node */
	node<P>* children;

	/** The number of children nodes of this node */
	unsigned short int num_children;

	/** Essentially, an upper bound on the distance to any child */
	short int scale;
};

template<class P>
void free_children(const node<P>& n)
{
	for (int i=0; i<n.num_children; i++)
	{
		free_children<P>(n.children[i]);
		n.children[i].~node<P>();
	}
	free(n.children);
}


/**
 * Cover tree node with an associated set of distances TODO better doc
 */
template<class P>
struct ds_node {

	ds_node() : dist(), p() {}

	/** Vector of distances TODO better doc*/
	v_array<ScalarType> dist;

	/** Point TODO better doc */
	P p;
};

static ScalarType base = COVERTREE_BASE;
static ScalarType il2 = 1. / log(base);

inline ScalarType dist_of_scale (int s)
{
	return pow(base, s);
}

inline int get_scale(ScalarType d)
{
	return (int)ceil(il2 * log(d));
}

	template<class P>
node<P> new_node(const P &p)
{
	node<P> new_node;
	new_node.p = p;
	return new_node;
}

	template<class P>
node<P> new_leaf(const P &p)
{
	node<P> new_leaf(p,0.,0.,NULL,0,100);
	return new_leaf;
}

	template<class P>
ScalarType max_set(v_array<ds_node<P> > &v)
{
	ScalarType max = 0.;
	for (int i = 0; i < v.index; i++)
		if ( max < v[i].dist.last())
			max = v[i].dist.last();
	return max;
}

void print_space(int s)
{
	for (int i = 0; i < s; i++)
		printf(" ");
}

template<class P>
void print(int depth, node<P> &top_node)
{
	print_space(depth);
	print(top_node.p);
	if ( top_node.num_children > 0 )
	{
		print_space(depth);
		printf("scale = %i\n",top_node.scale);
		print_space(depth);
		printf("max_dist = %f\n",top_node.max_dist);
		print_space(depth);
		printf("num children = %i\n",top_node.num_children);
		for (int i = 0; i < top_node.num_children;i++)
			print(depth+1, top_node.children[i]);
	}
}

template<class P>
void split(v_array<ds_node<P> >& point_set, v_array<ds_node<P> >& far_set, int max_scale)
{
	IndexType new_index = 0;
	ScalarType fmax = dist_of_scale(max_scale);
	for (int i = 0; i < point_set.index; i++)
	{
		if (point_set[i].dist.last() <= fmax)
		{
			point_set[new_index++] = point_set[i];
		}
		else
			push(far_set,point_set[i]);
	}
	point_set.index=new_index;
}

template<class P, class DistanceCallback>
void dist_split(DistanceCallback& dcb, v_array<ds_node<P> >& point_set,
		v_array<ds_node<P> >& new_point_set,
		P new_point,
		int max_scale)
{
	IndexType new_index = 0;
	ScalarType fmax = dist_of_scale(max_scale);
	for(int i = 0; i < point_set.index; i++)
	{
		ScalarType new_d;
		new_d = distance(dcb, new_point, point_set[i].p, fmax);
		if (new_d <= fmax )
		{
			push(point_set[i].dist, new_d);
			push(new_point_set,point_set[i]);
		}
		else
			point_set[new_index++] = point_set[i];
	}
	point_set.index = new_index;
}

/*
   max_scale is the maximum scale of the node we might create here.
   point_set contains points which are 2*max_scale or less away.
   */
template <class P, class DistanceCallback>
node<P> batch_insert(DistanceCallback& dcb, const P& p,
		int max_scale,
		int top_scale,
		v_array<ds_node<P> >& point_set,
		v_array<ds_node<P> >& consumed_set,
		v_array<v_array<ds_node<P> > >& stack)
{
	if (point_set.index == 0)
		return new_leaf(p);
	else {
		ScalarType max_dist = max_set(point_set); //O(|point_set|)
		int next_scale = std::min(max_scale - 1, get_scale(max_dist));
		if (next_scale == -2147483647-1) // We have points with distance 0.
		{
			v_array<node<P> > children;
			push(children,new_leaf(p));
			while (point_set.index > 0)
			{
				push(children,new_leaf(point_set.last().p));
				push(consumed_set,point_set.last());
				point_set.decr();
			}
			node<P> n = new_node(p);
			n.scale = 100; // A magic number meant to be larger than all scales.
			n.max_dist = 0;
			alloc(children,children.index);
			n.num_children = children.index;
			n.children = children.elements;
			return n;
		}
		else
		{
			v_array<ds_node<P> > far = pop(stack);
			split(point_set,far,max_scale); //O(|point_set|)

			node<P> child = batch_insert(dcb, p, next_scale, top_scale, point_set, consumed_set, stack);

			if (point_set.index == 0)
			{
				push(stack,point_set);
				point_set=far;
				return child;
			}
			else {
				node<P> n = new_node(p);
				v_array<node<P> > children;
				push(children, child);
				v_array<ds_node<P> > new_point_set = pop(stack);
				v_array<ds_node<P> > new_consumed_set = pop(stack);
				while (point_set.index != 0) { //O(|point_set| * num_children)
					P new_point = point_set.last().p;
					ScalarType new_dist = point_set.last().dist.last();
					push(consumed_set, point_set.last());
					point_set.decr();

					dist_split(dcb,point_set,new_point_set,new_point,max_scale); //O(|point_saet|)
					dist_split(dcb,far,new_point_set,new_point,max_scale); //O(|far|)

					node<P> new_child =
						batch_insert(dcb, new_point, next_scale, top_scale, new_point_set, new_consumed_set, stack);
					new_child.parent_dist = new_dist;

					push(children, new_child);

					ScalarType fmax = dist_of_scale(max_scale);
					for(int i = 0; i< new_point_set.index; i++) //O(|new_point_set|)
					{
						new_point_set[i].dist.decr();
						if (new_point_set[i].dist.last() <= fmax)
							push(point_set, new_point_set[i]);
						else
							push(far, new_point_set[i]);
					}
					for(int i = 0; i< new_consumed_set.index; i++) //O(|new_point_set|)
					{
						new_consumed_set[i].dist.decr();
						push(consumed_set, new_consumed_set[i]);
					}
					new_point_set.index = 0;
					new_consumed_set.index = 0;
				}
				push(stack,new_point_set);
				push(stack,new_consumed_set);
				push(stack,point_set);
				point_set=far;
				n.scale = top_scale - max_scale;
				n.max_dist = max_set(consumed_set);
				alloc(children,children.index);
				n.num_children = children.index;
				n.children = children.elements;
				return n;
			}
		}
	}
}

template<class P, class DistanceCallback>
node<P> batch_create(DistanceCallback& dcb, v_array<P> points)
{
	assert(points.index > 0);
	v_array<ds_node<P> > point_set;
	v_array<v_array<ds_node<P> > > stack;

	for (int i = 1; i < points.index; i++) {
		ds_node<P> temp;
		push(temp.dist, distance(dcb, points[0], points[i], std::numeric_limits<ScalarType>::max()));
		temp.p = points[i];
		push(point_set,temp);
	}

	v_array<ds_node<P> > consumed_set;

	ScalarType max_dist = max_set(point_set);

	node<P> top = batch_insert (dcb, points[0],
			get_scale(max_dist),
			get_scale(max_dist),
			point_set,
			consumed_set,
			stack);
	for (int i = 0; i<consumed_set.index;i++)
		free(consumed_set[i].dist.elements);
	free(consumed_set.elements);
	for (int i = 0; i<stack.index;i++)
		free(stack[i].elements);
	free(stack.elements);
	free(point_set.elements);
	return top;
}

void add_height(int d, v_array<int> &heights)
{
	if (heights.index <= d)
		for(;heights.index <= d;)
			push(heights,0);
	heights[d] = heights[d] + 1;
}

template <class P>
int height_dist(const node<P> top_node,v_array<int> &heights)
{
	if (top_node.num_children == 0)
	{
		add_height(0,heights);
		return 0;
	}
	else
	{
		int max_v=0;
		for (int i = 0; i<top_node.num_children ;i++)
		{
			int d = height_dist(top_node.children[i], heights);
			if (d > max_v)
				max_v = d;
		}
		add_height(1 + max_v, heights);
		return (1 + max_v);
	}
}

template <class P>
void depth_dist(int top_scale, const node<P> top_node,v_array<int> &depths)
{
	if (top_node.num_children > 0)
		for (int i = 0; i<top_node.num_children ;i++)
		{
			add_height(top_node.scale, depths);
			depth_dist(top_scale, top_node.children[i], depths);
		}
}

template <class P>
void breadth_dist(const node<P> top_node,v_array<int> &breadths)
{
	if (top_node.num_children == 0)
		add_height(0,breadths);
	else
	{
		for (int i = 0; i<top_node.num_children ;i++)
			breadth_dist(top_node.children[i], breadths);
		add_height(top_node.num_children, breadths);
	}
}

/**
 * List of cover tree nodes associated to a distance TODO better doc
 */
template <class P>
struct d_node
{
	/** Distance TODO better doc*/
	ScalarType dist;

	/** List of nodes TODO better doc*/
	const node<P> *n;
};

template <class P>
inline ScalarType compare(const d_node<P> *p1, const d_node<P>* p2)
{
	return p1 -> dist - p2 -> dist;
}

template <class P>
void halfsort (v_array<d_node<P> > cover_set)
{
	if (cover_set.index <= 1)
		return;
	register d_node<P> *base_ptr =  cover_set.elements;

	d_node<P> *hi = &base_ptr[cover_set.index - 1];
	d_node<P> *right_ptr = hi;
	d_node<P> *left_ptr;

	while (right_ptr > base_ptr)
	{
		d_node<P> *mid = base_ptr + ((hi - base_ptr) >> 1);

		if (compare ( mid,  base_ptr) < 0.)
			std::swap(*mid, *base_ptr);
		if (compare ( hi,  mid) < 0.)
			std::swap(*mid, *hi);
		else
			goto jump_over;
		if (compare ( mid,  base_ptr) < 0.)
			std::swap(*mid, *base_ptr);
jump_over:;

		left_ptr  = base_ptr + 1;
		right_ptr = hi - 1;

		do
		{
			while (compare (left_ptr, mid) < 0.)
				left_ptr++;

			while (compare (mid, right_ptr) < 0.)
				right_ptr--;

			if (left_ptr < right_ptr)
			{
				std::swap(*left_ptr, *right_ptr);
				if (mid == left_ptr)
					mid = right_ptr;
				else if (mid == right_ptr)
					mid = left_ptr;
				left_ptr++;
				right_ptr--;
			}
			else if (left_ptr == right_ptr)
			{
				left_ptr ++;
				right_ptr --;
				break;
			}
		}
		while (left_ptr <= right_ptr);
		hi = right_ptr;
	}
}

template <class P>
v_array<v_array<d_node<P> > > get_cover_sets(v_array<v_array<v_array<d_node<P> > > > &spare_cover_sets)
{
	v_array<v_array<d_node<P> > > ret = pop(spare_cover_sets);
	while (ret.index < 101)
	{
		v_array<d_node<P> > temp;
		push(ret, temp);
	}
	return ret;
}

inline bool shell(ScalarType parent_query_dist, ScalarType child_parent_dist, ScalarType upper_bound)
{
	return parent_query_dist - child_parent_dist <= upper_bound;
	//    && child_parent_dist - parent_query_dist <= upper_bound;
}

int internal_k =1;
void update_k(ScalarType *k_upper_bound, ScalarType upper_bound)
{
	ScalarType *end = k_upper_bound + internal_k-1;
	ScalarType *begin = k_upper_bound;
	for (;end != begin; begin++)
	{
		if (upper_bound < *(begin+1))
			*begin = *(begin+1);
		else {
			*begin = upper_bound;
			break;
		}
	}
	if (end == begin)
		*begin = upper_bound;
}
ScalarType *alloc_k()
{
	return (ScalarType*)malloc(sizeof(ScalarType) * internal_k);
}
void set_k(ScalarType* begin, ScalarType max)
{
	for(ScalarType *end = begin+internal_k;end != begin; begin++)
		*begin = max;
}

ScalarType internal_epsilon =0.;
//void update_epsilon(ScalarType *upper_bound, ScalarType new_dist) {}
ScalarType *alloc_epsilon()
{
	return (ScalarType *)malloc(sizeof(ScalarType));
}
void set_epsilon(ScalarType* begin)
{
	*begin = internal_epsilon;
}

void update_unequal(ScalarType *upper_bound, ScalarType new_dist)
{
	if (new_dist != 0.)
		*upper_bound = new_dist;
}
ScalarType* (*alloc_unequal)() = alloc_epsilon;
void set_unequal(ScalarType* begin, ScalarType max)
{
	*begin = max;
}

void (*update)(ScalarType *foo, ScalarType bar) = update_k;
void (*setter)(ScalarType *foo, ScalarType bar) = set_k;
ScalarType* (*alloc_upper)() = alloc_k;

template <class P, class DistanceCallback>
inline void copy_zero_set(DistanceCallback& dcb, node<P>* query_chi,
		ScalarType* new_upper_bound, v_array<d_node<P> > &zero_set,
		v_array<d_node<P> > &new_zero_set)
{
	new_zero_set.index = 0;
	d_node<P> *end = zero_set.elements + zero_set.index;
	for (d_node<P> *ele = zero_set.elements; ele != end ; ele++)
	{
		ScalarType upper_dist = *new_upper_bound + query_chi->max_dist;
		if (shell(ele->dist, query_chi->parent_dist, upper_dist))
		{
			ScalarType d = distance(dcb, query_chi->p, ele->n->p, upper_dist);

			if (d <= upper_dist)
			{
				if (d < *new_upper_bound)
					update(new_upper_bound, d);
				d_node<P> temp = {d, ele->n};
				push(new_zero_set,temp);
			}
		}
	}
}

template <class P, class DistanceCallback>
inline void copy_cover_sets(DistanceCallback& dcb, node<P>* query_chi,
		ScalarType* new_upper_bound,
		v_array<v_array<d_node<P> > > &cover_sets,
		v_array<v_array<d_node<P> > > &new_cover_sets,
		int current_scale, int max_scale)
{
	for (; current_scale <= max_scale; current_scale++)
	{
		d_node<P>* ele = cover_sets[current_scale].elements;
		d_node<P>* end = cover_sets[current_scale].elements + cover_sets[current_scale].index;
		for (; ele != end; ele++)
		{
			ScalarType upper_dist = *new_upper_bound + query_chi->max_dist + ele->n->max_dist;
			if (shell(ele->dist, query_chi->parent_dist, upper_dist))
			{
				ScalarType d = distance(dcb, query_chi->p, ele->n->p, upper_dist);

				if (d <= upper_dist)
				{
					if (d < *new_upper_bound)
						update(new_upper_bound,d);
					d_node<P> temp = {d, ele->n};
					push(new_cover_sets[current_scale],temp);
				}
			}
		}
	}
}

template <class P>
void print_query(const node<P> *top_node)
{
	printf("query = \n");
	print(top_node->p);
	if ( top_node->num_children > 0 ) {
		printf("scale = %i\n",top_node->scale);
		printf("max_dist = %f\n",top_node->max_dist);
		printf("num children = %i\n",top_node->num_children);
	}
}

template <class P>
void print_cover_sets(v_array<v_array<d_node<P> > > &cover_sets,
		v_array<d_node<P> > &zero_set,
		int current_scale, int max_scale)
{
	printf("cover set = \n");
	for (; current_scale <= max_scale; current_scale++)
	{
		d_node<P> *ele = cover_sets[current_scale].elements;
		d_node<P> *end = cover_sets[current_scale].elements + cover_sets[current_scale].index;
		printf("%i\n", current_scale);
		for (; ele != end; ele++)
		{
			node<P> *n = (node<P> *)ele->n;
			print(n->p);
		}
	}
	d_node<P> *end = zero_set.elements + zero_set.index;
	printf("infinity\n");
	for (d_node<P> *ele = zero_set.elements; ele != end ; ele++)
	{
		node<P> *n = (node<P> *)ele->n;
		print(n->p);
	}
}

/*
   An optimization to consider:
   Make all distance evaluations occur in descend.

   Instead of passing a cover_set, pass a stack of cover sets.  The
   last element holds d_nodes with your distance.  The next lower
   element holds a d_node with the distance to your query parent,
   next = query grand parent, etc..

   Compute distances in the presence of the tighter upper bound.
   */
template <class P, class DistanceCallback>
inline
void descend(DistanceCallback& dcb, const node<P>* query, ScalarType* upper_bound,
		int current_scale,int &max_scale, v_array<v_array<d_node<P> > > &cover_sets,
		v_array<d_node<P> > &zero_set)
{
	d_node<P> *end = cover_sets[current_scale].elements + cover_sets[current_scale].index;
	for (d_node<P> *parent = cover_sets[current_scale].elements; parent != end; parent++)
	{
		const node<P> *par = parent->n;
		ScalarType upper_dist = *upper_bound + query->max_dist + query->max_dist;
		if (parent->dist <= upper_dist + par->max_dist)
		{
			node<P> *chi = par->children;
			if (parent->dist <= upper_dist + chi->max_dist)
			{
				if (chi->num_children > 0)
				{
					if (max_scale < chi->scale)
						max_scale = chi->scale;
					d_node<P> temp = {parent->dist, chi};
					push(cover_sets[chi->scale], temp);
				}
				else if (parent->dist <= upper_dist)
				{
					d_node<P> temp = {parent->dist, chi};
					push(zero_set, temp);
				}
			}
			node<P> *child_end = par->children + par->num_children;
			for (chi++; chi != child_end; chi++)
			{
				ScalarType upper_chi = *upper_bound + chi->max_dist + query->max_dist + query->max_dist;
				if (shell(parent->dist, chi->parent_dist, upper_chi))
				{
					ScalarType d = distance(dcb, query->p, chi->p, upper_chi);
					if (d <= upper_chi)
					{
						if (d < *upper_bound)
							update(upper_bound, d);
						if (chi->num_children > 0)
						{
							if (max_scale < chi->scale)
								max_scale = chi->scale;
							d_node<P> temp = {d, chi};
							push(cover_sets[chi->scale],temp);
						}
						else
							if (d <= upper_chi - chi->max_dist)
							{
								d_node<P> temp = {d, chi};
								push(zero_set, temp);
							}
					}
				}
			}
		}
	}
}

template <class P, class DistanceCallback>
void brute_nearest(DistanceCallback& dcb, const node<P>* query,
		v_array<d_node<P> > zero_set, ScalarType* upper_bound,
		v_array<v_array<P> > &results,
		v_array<v_array<d_node<P> > > &spare_zero_sets)
{
	if (query->num_children > 0)
	{
		v_array<d_node<P> > new_zero_set = pop(spare_zero_sets);
		node<P> * query_chi = query->children;
		brute_nearest(dcb, query_chi, zero_set, upper_bound, results, spare_zero_sets);
		ScalarType* new_upper_bound = alloc_upper();

		node<P> *child_end = query->children + query->num_children;
		for (query_chi++;query_chi != child_end; query_chi++)
		{
			setter(new_upper_bound,*upper_bound + query_chi->parent_dist);
			copy_zero_set(dcb, query_chi, new_upper_bound, zero_set, new_zero_set);
			brute_nearest(dcb, query_chi, new_zero_set, new_upper_bound, results, spare_zero_sets);
		}
		free (new_upper_bound);
		new_zero_set.index = 0;
		push(spare_zero_sets, new_zero_set);
	}
	else
	{
		v_array<P> temp;
		push(temp, query->p);
		d_node<P> *end = zero_set.elements + zero_set.index;
		for (d_node<P> *ele = zero_set.elements; ele != end ; ele++)
			if (ele->dist <= *upper_bound)
				push(temp, ele->n->p);
		push(results,temp);
	}
}

template <class P, class DistanceCallback>
void internal_batch_nearest_neighbor(DistanceCallback& dcb, const node<P> *query,
		v_array<v_array<d_node<P> > > &cover_sets,
		v_array<d_node<P> > &zero_set,
		int current_scale,
		int max_scale,
		ScalarType* upper_bound,
		v_array<v_array<P> > &results,
		v_array<v_array<v_array<d_node<P> > > > &spare_cover_sets,
		v_array<v_array<d_node<P> > > &spare_zero_sets)
{
	if (current_scale > max_scale) // All remaining points are in the zero set.
		brute_nearest(dcb, query, zero_set, upper_bound, results, spare_zero_sets);
	else
		if (query->scale <= current_scale && query->scale != 100)
			// Our query has too much scale.  Reduce.
		{
			node<P> *query_chi = query->children;
			v_array<d_node<P> > new_zero_set = pop(spare_zero_sets);
			v_array<v_array<d_node<P> > > new_cover_sets = get_cover_sets(spare_cover_sets);
			ScalarType* new_upper_bound = alloc_upper();

			node<P> *child_end = query->children + query->num_children;
			for (query_chi++; query_chi != child_end; query_chi++)
			{
				setter(new_upper_bound,*upper_bound + query_chi->parent_dist);
				copy_zero_set(dcb, query_chi, new_upper_bound, zero_set, new_zero_set);
				copy_cover_sets(dcb, query_chi, new_upper_bound, cover_sets, new_cover_sets,
						current_scale, max_scale);
				internal_batch_nearest_neighbor(dcb, query_chi, new_cover_sets, new_zero_set,
						current_scale, max_scale, new_upper_bound,
						results, spare_cover_sets, spare_zero_sets);
			}
			free (new_upper_bound);
			new_zero_set.index = 0;
			push(spare_zero_sets, new_zero_set);
			push(spare_cover_sets, new_cover_sets);
			internal_batch_nearest_neighbor(dcb, query->children, cover_sets, zero_set,
					current_scale, max_scale, upper_bound, results,
					spare_cover_sets, spare_zero_sets);
		}
		else // reduce cover set scale
		{
			halfsort(cover_sets[current_scale]);
			descend(dcb, query, upper_bound, current_scale, max_scale,cover_sets, zero_set);
			cover_sets[current_scale++].index = 0;
			internal_batch_nearest_neighbor(dcb, query, cover_sets, zero_set,
					current_scale, max_scale, upper_bound, results,
					spare_cover_sets, spare_zero_sets);
		}
}

template <class P, class DistanceCallback>
void batch_nearest_neighbor(DistanceCallback &dcb, const node<P> &top_node,
		const node<P> &query, v_array<v_array<P> > &results)
{
	v_array<v_array<v_array<d_node<P> > > > spare_cover_sets;
	v_array<v_array<d_node<P> > > spare_zero_sets;

	v_array<v_array<d_node<P> > > cover_sets = get_cover_sets(spare_cover_sets);
	v_array<d_node<P> > zero_set = pop(spare_zero_sets);

	ScalarType* upper_bound = alloc_upper();
	setter(upper_bound, std::numeric_limits<ScalarType>::max());

	ScalarType top_dist = distance(dcb, query.p, top_node.p, std::numeric_limits<ScalarType>::max());
	update(upper_bound, top_dist);

	d_node<P> temp = {top_dist, &top_node};
	push(cover_sets[0], temp);

	internal_batch_nearest_neighbor(dcb, &query,cover_sets,zero_set,0,0,upper_bound,results,
			spare_cover_sets,spare_zero_sets);

	free(upper_bound);
	push(spare_cover_sets, cover_sets);

	for (int i = 0; i < spare_cover_sets.index; i++)
	{
		v_array<v_array<d_node<P> > > cover_sets2 = spare_cover_sets[i];
		for (int j = 0; j < cover_sets2.index; j++)
			free (cover_sets2[j].elements);
		free(cover_sets2.elements);
	}
	free(spare_cover_sets.elements);

	push(spare_zero_sets, zero_set);

	for (int i = 0; i < spare_zero_sets.index; i++)
		free(spare_zero_sets[i].elements);
	free(spare_zero_sets.elements);
}

template <class P, class DistanceCallback>
void k_nearest_neighbor(DistanceCallback &dcb, const node<P> &top_node,
		const node<P> &query, v_array<v_array<P> > &results, int k)
{
	internal_k = k;
	update = update_k;
	setter = set_k;
	alloc_upper = alloc_k;

	batch_nearest_neighbor(dcb, top_node, query, results);
}
/*
template <class P, class DistanceCallback>
void epsilon_nearest_neighbor(DistanceCallback &dcb, const node<P> &top_node,
		const node<P> &query, v_array<v_array<P> > &results,
		ScalarType epsilon)
{
	internal_epsilon = epsilon;
	//  update = update_epsilon;
	setter = set_epsilon;
	alloc_upper = alloc_epsilon;

	batch_nearest_neighbor(dcb, top_node, query, results);
}

template <class P, class DistanceCallback>
void unequal_nearest_neighbor(DistanceCallback &dcb, const node<P> &top_node,
		const node<P> &query, v_array<v_array<P> > &results)
{
	update = update_unequal;
	setter = set_unequal;
	alloc_upper = alloc_unequal;

	batch_nearest_neighbor(dcb, top_node, query, results);
}
*/

}
}
#endif
