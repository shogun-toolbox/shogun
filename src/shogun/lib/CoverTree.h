/*
 * This program is SG_FREE software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the SG_FREE Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * First written by John Langford jl@hunch.net
 * Templatization by Dinoj Surendran dinojs@gmail.com
 *
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

/* Whatever structure/class/type is used for P, it must have the following methods defined:

  static float64_t P::distance(const P &p1, const P &p2, float64_t upper_bound)
  : this returns the distance between two P objects
  : the distance does not have to be calculated fully if it's more than upper_bound

  static void P::print(const P &p)
  : this prints out the contents of a P object.
*/

#ifndef COVERTREE_H
#define COVERTREE_H

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

#include <stdio.h>

#include <assert.h>
#include <values.h>

#include <shogun/lib/v_array.h>

namespace shogun
{

template<class T> void push(v_array<T>& v, const T &new_ele)
{
    if(v.end == v.end_array)
    {
        size_t old_length = v.end_array - v.begin;
        size_t new_length = 2 * old_length + 3;

        v.begin = SG_REALLOC(T, v.begin, new_length);
        v.end = v.begin + old_length;
        v.end_array = v.begin + new_length;
    }
    *(v.end++) = new_ele;
}

template<class T> void alloc(v_array<T>& v, int32_t length)
{
    int32_t index = v.index();
    v.begin = SG_REALLOC(T, v.begin, length);

    v.end = v.begin + index;
    v.end_array = v.begin + length;
}

template<class T> v_array<T> pop(v_array<v_array<T> > &stack)
{
    if (stack.end != stack.begin)
        return *(--stack.end);
    else
        return v_array<T>();
}

template<class P> struct node
{
    P p;
    float64_t max_dist;  // The maximum distance to any grandchild.
    float64_t parent_dist; // The distance to the parent.
    node<P>* children;
    int32_t num_children; // The number of children.
    int32_t scale; // Essentially, an upper bound on the distance to any child.
};


template<class P> struct ds_node
{
    v_array<float64_t> dist;
    P p;
};

float64_t base = 1.3;
float64_t il2 = 1. / CMath::log(base);

inline float64_t dist_of_scale (int32_t s)
{
    return pow(base, s);
}

inline int32_t get_scale(float64_t d)
{
    return (int32_t) ceilf(il2 * log(d));
}

int32_t min(int32_t f1, int32_t f2)
{
    if ( f1 <= f2 )
        return f1;
    else
        return f2;
}

float64_t max(float64_t f1, float64_t f2)
{
    if ( f1 <= f2 )
        return f2;
    else
        return f1;
}

template<class P> node<P> new_node(const P &p)
{
    node<P> new_node;
    new_node.p = p;
    return new_node;
}

template<class P> node<P> new_leaf(const P &p)
{
    node<P> new_leaf = {p,0.,0.,NULL,0,100};
    return new_leaf;
}

template<class P> float64_t max_set(v_array<ds_node<P> > &v)
{
    float64_t max = 0.;
    for (int32_t i = 0; i < v.index(); i++)
        if ( max < v[i].dist.last())
            max = v[i].dist.last();
    return max;
}

void print_space(int32_t s)
{
    for (int32_t i = 0; i < s; i++)
        printf(" ");
}

template<class P> void print(int32_t depth, node<P> &top_node)
{
    print_space(depth);
    P::print(top_node.p);
    if ( top_node.num_children > 0 )
    {
        print_space(depth);
        printf("scale = %i\n",top_node.scale);
        print_space(depth);
        printf("max_dist = %f\n",top_node.max_dist);
        print_space(depth);
        printf("num children = %i\n",top_node.num_children);
        for (int32_t i = 0; i < top_node.num_children; i++)
            print(depth+1, top_node.children[i]);
    }
}

template<class P> void split(v_array<ds_node<P> >& point_set, v_array<ds_node<P> >& far_set, int32_t max_scale)
{
    int32_t new_index = 0;
    float64_t fmax = dist_of_scale(max_scale);
    for (int32_t i = 0; i < point_set.index(); i++)
    {
        if (point_set[i].dist.last() <= fmax)
        {
            point_set[new_index++] = point_set[i];
        }
        else
            push(far_set,point_set[i]);
    }
    //point_set.index=new_index;
    point_set.end = point_set.begin + new_index;
}

template<class P> void dist_split(v_array<ds_node<P> >& point_set,
                                  v_array<ds_node<P> >& new_point_set,
                                  P new_point,
                                  int32_t max_scale)
{
    int32_t new_index = 0;
    float64_t fmax = dist_of_scale(max_scale);
    for(int32_t i = 0; i < point_set.index(); i++)
    {
        float64_t new_d;
        new_d = P::distance(new_point, point_set[i].p, fmax);
        if (new_d <= fmax )
        {
            push(point_set[i].dist, new_d);
            push(new_point_set,point_set[i]);
        }
        else
            point_set[new_index++] = point_set[i];
    }
    //point_set.index = new_index;
    point_set.end = point_set.begin + new_index;
}

/*
   max_scale is the maximum scale of the node we might create here.
   point_set contains points which are 2*max_scale or less away.
*/

template <class P> node<P> batch_insert(const P& p,
                                        int32_t max_scale,
                                        int32_t top_scale,
                                        v_array<ds_node<P> >& point_set,
                                        v_array<ds_node<P> >& consumed_set,
                                        v_array<v_array<ds_node<P> > >& stack)
{
    if (point_set.index() == 0)
        return new_leaf(p);
    else
    {
        float64_t max_dist = max_set(point_set); //O(|point_set|)
        int32_t next_scale = min (max_scale - 1, get_scale(max_dist));
        if (next_scale == -2147483647-1) // We have points with distance 0.
        {
            v_array<node<P> > children;
            push(children,new_leaf(p));
            while (point_set.index() > 0)
            {
                push(children,new_leaf(point_set.last().p));
                push(consumed_set,point_set.last());
                point_set.decr();
            }
            node<P> n = new_node(p);
            n.scale = 100; // A magic number meant to be larger than all scales.
            n.max_dist = 0;
            alloc(children,children.index());
            n.num_children = children.index();
            n.children = children.begin;
            return n;
        }
        else
        {
            v_array<ds_node<P> > far = pop(stack);
            split(point_set,far,max_scale); //O(|point_set|)

            node<P> child = batch_insert(p, next_scale, top_scale, point_set, consumed_set, stack);

            if (point_set.index() == 0)
            {
                push(stack,point_set);
                point_set=far;
                return child;
            }
            else
            {
                node<P> n = new_node(p);
                v_array<node<P> > children;
                push(children, child);
                v_array<ds_node<P> > new_point_set = pop(stack);
                v_array<ds_node<P> > new_consumed_set = pop(stack);
                while (point_set.index() != 0)   //O(|point_set| * num_children)
                {
                    P new_point = point_set.last().p;
                    float64_t new_dist = point_set.last().dist.last();
                    push(consumed_set, point_set.last());
                    point_set.decr();

                    dist_split(point_set, new_point_set, new_point, max_scale); //O(|point_saet|)
                    dist_split(far,new_point_set,new_point,max_scale); //O(|far|)

                    node<P> new_child =
                        batch_insert(new_point, next_scale, top_scale, new_point_set, new_consumed_set, stack);
                    new_child.parent_dist = new_dist;

                    push(children, new_child);

                    float64_t fmax = dist_of_scale(max_scale);
                    for(int32_t i = 0; i< new_point_set.index(); i++) //O(|new_point_set|)
                    {
                        new_point_set[i].dist.decr();
                        if (new_point_set[i].dist.last() <= fmax)
                            push(point_set, new_point_set[i]);
                        else
                            push(far, new_point_set[i]);
                    }
                    for(int32_t i = 0; i< new_consumed_set.index(); i++) //O(|new_point_set|)
                    {
                        new_consumed_set[i].dist.decr();
                        push(consumed_set, new_consumed_set[i]);
                    }
                    //new_point_set.index = 0;
                    //new_consumed_set.index = 0;
                    new_point_set.erase();
                    new_consumed_set.erase();
                }
                push(stack,new_point_set);
                push(stack,new_consumed_set);
                push(stack,point_set);
                point_set=far;
                n.scale = top_scale - max_scale;
                n.max_dist = max_set(consumed_set);
                alloc(children,children.index());
                n.num_children = children.index();
                n.children = children.begin;
                return n;
            }
        }
    }
}

template<class P> node<P> batch_create(v_array<P> points)
{
    assert(points.index() > 0);
    v_array<ds_node<P> > point_set;
    v_array<v_array<ds_node<P> > > stack;

    for (int32_t i = 1; i < points.index(); i++)
    {
        ds_node<P> temp;
        push(temp.dist, P::distance(points[0], points[i], MAXFLOAT));
        temp.p = points[i];
        push(point_set,temp);
    }

    v_array<ds_node<P> > consumed_set;

    float64_t max_dist = max_set(point_set);

    node<P> top = batch_insert (points[0],
                                get_scale(max_dist),
                                get_scale(max_dist),
                                point_set,
                                consumed_set,
                                stack);
    for (int32_t i = 0; i<consumed_set.index(); i++)
        SG_FREE(consumed_set[i].dist.begin);
    SG_FREE(consumed_set.begin);
    for (int32_t i = 0; i<stack.index(); i++)
        SG_FREE(stack[i].begin);
    SG_FREE(stack.begin);
    SG_FREE(point_set.begin);
    return top;
}

void add_height(int32_t d, v_array<int32_t> &heights)
{
    if (heights.index() <= d)
        for(; heights.index() <= d;)
            push(heights,0);
    heights[d] = heights[d] + 1;
}

template <class P> int32_t height_dist(const node<P> top_node,v_array<int32_t> &heights)
{
    if (top_node.num_children == 0)
    {
        add_height(0,heights);
        return 0;
    }
    else
    {
        int32_t max_v=0;
        for (int32_t i = 0; i<top_node.num_children ; i++)
        {
            int32_t d = height_dist(top_node.children[i], heights);
            if (d > max_v)
                max_v = d;
        }
        add_height(1 + max_v, heights);
        return (1 + max_v);
    }
}

template <class P> void depth_dist(int32_t top_scale, const node<P> top_node,v_array<int32_t> &depths)
{
    if (top_node.num_children > 0)
        for (int32_t i = 0; i<top_node.num_children ; i++)
        {
            add_height(top_node.scale, depths);
            depth_dist(top_scale, top_node.children[i], depths);
        }
}

template <class P> void breadth_dist(const node<P> top_node,v_array<int32_t> &breadths)
{
    if (top_node.num_children == 0)
        add_height(0,breadths);
    else
    {
        for (int32_t i = 0; i<top_node.num_children ; i++)
            breadth_dist(top_node.children[i], breadths);
        add_height(top_node.num_children, breadths);
    }
}

template <class P> struct d_node
{
    float64_t dist;
    const node<P> *n;
};

template <class P> inline float64_t compare(const d_node<P> *p1, const d_node<P>* p2)
{
    return p1 -> dist - p2 -> dist;
}

template <class P> void SWAP (d_node<P>* a, d_node<P>* b)
{
    d_node<P> tmp = *a;
    *a = *b;
    *b = tmp;
}

template <class P> void halfsort (v_array<d_node<P> > cover_set)
{
    if (cover_set.index() <= 1)
        return;
    register d_node<P> *base_ptr =  cover_set.begin;

    d_node<P> *hi = &base_ptr[cover_set.index() - 1];
    d_node<P> *right_ptr = hi;
    d_node<P> *left_ptr;

    while (right_ptr > base_ptr)
    {
        d_node<P> *mid = base_ptr + ((hi - base_ptr) >> 1);

        if (compare ( mid,  base_ptr) < 0.)
            SWAP (mid, base_ptr);
        if (compare ( hi,  mid) < 0.)
            SWAP (mid, hi);
        else
            goto jump_over;
        if (compare ( mid,  base_ptr) < 0.)
            SWAP (mid, base_ptr);
jump_over:
        ;

        left_ptr  = base_ptr + 1;
        right_ptr = hi - 1;

        do
        {
            while (compare (left_ptr, mid) < 0.)
                left_ptr ++;

            while (compare (mid, right_ptr) < 0.)
                right_ptr --;

            if (left_ptr < right_ptr)
            {
                SWAP (left_ptr, right_ptr);
                if (mid == left_ptr)
                    mid = right_ptr;
                else if (mid == right_ptr)
                    mid = left_ptr;
                left_ptr ++;
                right_ptr --;
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

template <class P> v_array<v_array<d_node<P> > > get_cover_sets(v_array<v_array<v_array<d_node<P> > > > &spare_cover_sets)
{
    v_array<v_array<d_node<P> > > ret = pop(spare_cover_sets);
    while (ret.index() < 101)
    {
        v_array<d_node<P> > temp;
        push(ret, temp);
    }
    return ret;
}

inline bool shell(float64_t parent_query_dist, float64_t child_parent_dist, float64_t upper_bound)
{
    return parent_query_dist - child_parent_dist <= upper_bound;
}

int32_t internal_k = 1;

void update_k(float64_t *k_upper_bound, float64_t upper_bound)
{
    float64_t *end = k_upper_bound + internal_k-1;
    float64_t *begin = k_upper_bound;
    for (; end != begin; begin++)
    {
        if (upper_bound < *(begin+1))
            *begin = *(begin+1);
        else
        {
            *begin = upper_bound;
            break;
        }
    }
    if (end == begin)
        *begin = upper_bound;
}
float64_t *alloc_k()
{
    return (float64_t *)malloc(sizeof(float64_t) * internal_k);
}
void set_k(float64_t* begin, float64_t max)
{
    for(float64_t *end = begin+internal_k; end != begin; begin++)
        *begin = max;
}

float64_t internal_epsilon =0.;
void update_epsilon(float64_t *upper_bound, float64_t new_dist) {}
float64_t *alloc_epsilon()
{
    return (float64_t *)malloc(sizeof(float64_t));
}
void set_epsilon(float64_t* begin, float64_t max)
{
    *begin = internal_epsilon;
}

void update_unequal(float64_t *upper_bound, float64_t new_dist)
{
    if (new_dist != 0.)
        *upper_bound = new_dist;
}
float64_t* (*alloc_unequal)() = alloc_epsilon;
void set_unequal(float64_t* begin, float64_t max)
{
    *begin = max;
}

void (*update)(float64_t *foo, float64_t bar) = update_k;
void (*setter)(float64_t *foo, float64_t bar) = set_k;
float64_t* (*alloc_upper)() = alloc_k;

template <class P> inline void copy_zero_set(node<P>* query_chi, float64_t* new_upper_bound,
        v_array<d_node<P> > &zero_set, v_array<d_node<P> > &new_zero_set)
{
    new_zero_set.erase();
    d_node<P> *end = zero_set.begin + zero_set.index();
    for (d_node<P> *ele = zero_set.begin; ele != end ; ele++)
    {
        float64_t upper_dist = *new_upper_bound + query_chi->max_dist;
        if (shell(ele->dist, query_chi->parent_dist, upper_dist))
        {
            float64_t d = P::distance(query_chi->p, ele->n->p, upper_dist);

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

template <class P> inline void copy_cover_sets(node<P>* query_chi, float64_t* new_upper_bound,
        v_array<v_array<d_node<P> > > &cover_sets,
        v_array<v_array<d_node<P> > > &new_cover_sets,
        int32_t current_scale, int32_t max_scale)
{
    for (; current_scale <= max_scale; current_scale++)
    {
        d_node<P>* ele = cover_sets[current_scale].begin;
        d_node<P>* end = cover_sets[current_scale].begin + cover_sets[current_scale].index();
        for (; ele != end; ele++)
        {
            float64_t upper_dist = *new_upper_bound + query_chi->max_dist + ele->n->max_dist;
            if (shell(ele->dist, query_chi->parent_dist, upper_dist))
            {
                float64_t d = P::distance(query_chi->p, ele->n->p, upper_dist);

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

template <class P> void print_query(const node<P> *top_node)
{
    printf ("query = \n");
    P::print(top_node->p);
    if ( top_node->num_children > 0 )
    {
        printf("scale = %i\n",top_node->scale);
        printf("max_dist = %f\n",top_node->max_dist);
        printf("num children = %i\n",top_node->num_children);
    }
}

template <class P> void print_cover_sets(v_array<v_array<d_node<P> > > &cover_sets,
        v_array<d_node<P> > &zero_set,
        int32_t current_scale, int32_t max_scale)
{
    printf("cover set = \n");
    for (; current_scale <= max_scale; current_scale++)
    {
        d_node<P> *ele = cover_sets[current_scale].begin;
        d_node<P> *end = cover_sets[current_scale].begin + cover_sets[current_scale].index();
        printf ("%i\n", current_scale);
        for (; ele != end; ele++)
        {
            node<P> *n = (node<P> *)ele->n;
            P::print(n->p);
        }
    }
    d_node<P> *end = zero_set.begin + zero_set.index();
    printf ("infinity\n");
    for (d_node<P> *ele = zero_set.begin; ele != end ; ele++)
    {
        node<P> *n = (node<P> *)ele->n;
        P::print(n->p);
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

template <class P> inline void descend(const node<P>* query, float64_t* upper_bound,
                                       int32_t current_scale,
                                       int32_t &max_scale, v_array<v_array<d_node<P> > > &cover_sets,
                                       v_array<d_node<P> > &zero_set)
{
    d_node<P> *end = cover_sets[current_scale].begin + cover_sets[current_scale].index();
    for (d_node<P> *parent = cover_sets[current_scale].begin; parent != end; parent++)
    {
        const node<P> *par = parent->n;
        float64_t upper_dist = *upper_bound + query->max_dist + query->max_dist;
        if (parent->dist <= upper_dist + par->max_dist)
        {
            node<P> *chi = par->children;
            if (parent->dist <= upper_dist + chi->max_dist)
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
            node<P> *child_end = par->children + par->num_children;
            for (chi++; chi != child_end; chi++)
            {
                float64_t upper_chi = *upper_bound + chi->max_dist + query->max_dist + query->max_dist;
                if (shell(parent->dist, chi->parent_dist, upper_chi))
                {
                    float64_t d = P::distance(query->p, chi->p, upper_chi);
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
                        else if (d <= upper_chi - chi->max_dist)
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

template <class P> void brute_nearest(const node<P>* query,v_array<d_node<P> > zero_set,
                                      float64_t* upper_bound,
                                      v_array<v_array<P> > &results,
                                      v_array<v_array<d_node<P> > > &spare_zero_sets)
{
    if (query->num_children > 0)
    {
        v_array<d_node<P> > new_zero_set = pop(spare_zero_sets);
        node<P> * query_chi = query->children;
        brute_nearest(query_chi, zero_set, upper_bound, results, spare_zero_sets);
        float64_t* new_upper_bound = alloc_upper();

        node<P> *child_end = query->children + query->num_children;
        for (query_chi++; query_chi != child_end; query_chi++)
        {
            setter(new_upper_bound,*upper_bound + query_chi->parent_dist);
            copy_zero_set(query_chi, new_upper_bound, zero_set, new_zero_set);
            brute_nearest(query_chi, new_zero_set, new_upper_bound, results, spare_zero_sets);
        }
        SG_FREE (new_upper_bound);
        new_zero_set.erase();
        push(spare_zero_sets, new_zero_set);
    }
    else
    {
        v_array<P> temp;
        push(temp, query->p);
        d_node<P> *end = zero_set.begin + zero_set.index();
        for (d_node<P> *ele = zero_set.begin; ele != end ; ele++)
            if (ele->dist <= *upper_bound)
                push(temp, ele->n->p);
        push(results,temp);
    }
}

template <class P> void internal_batch_nearest_neighbor(const node<P> *query,
        v_array<v_array<d_node<P> > > &cover_sets,
        v_array<d_node<P> > &zero_set,
        int32_t current_scale,
        int32_t max_scale,
        float64_t* upper_bound,
        v_array<v_array<P> > &results,
        v_array<v_array<v_array<d_node<P> > > > &spare_cover_sets,
        v_array<v_array<d_node<P> > > &spare_zero_sets)
{
    if (current_scale > max_scale) // All remaining points are in the zero set.
        brute_nearest(query, zero_set, upper_bound, results, spare_zero_sets);
    else if (query->scale <= current_scale && query->scale != 100)
        // Our query has too much scale.  Reduce.
    {
        node<P> *query_chi = query->children;
        v_array<d_node<P> > new_zero_set = pop(spare_zero_sets);
        v_array<v_array<d_node<P> > > new_cover_sets = get_cover_sets(spare_cover_sets);
        float64_t* new_upper_bound = alloc_upper();

        node<P> *child_end = query->children + query->num_children;
        for (query_chi++; query_chi != child_end; query_chi++)
        {
            setter(new_upper_bound,*upper_bound + query_chi->parent_dist);
            copy_zero_set(query_chi, new_upper_bound, zero_set, new_zero_set);
            copy_cover_sets(query_chi, new_upper_bound, cover_sets, new_cover_sets,
                            current_scale, max_scale);
            internal_batch_nearest_neighbor(query_chi, new_cover_sets, new_zero_set,
                                            current_scale, max_scale, new_upper_bound,
                                            results, spare_cover_sets, spare_zero_sets);
        }
        SG_FREE (new_upper_bound);

        new_zero_set.erase();
        push(spare_zero_sets, new_zero_set);
        push(spare_cover_sets, new_cover_sets);
        internal_batch_nearest_neighbor(query->children, cover_sets, zero_set,
                                        current_scale, max_scale, upper_bound, results,
                                        spare_cover_sets, spare_zero_sets);
    }
    else // reduce cover set scale
    {
        halfsort(cover_sets[current_scale]);
        descend(query, upper_bound, current_scale, max_scale,cover_sets, zero_set);

        cover_sets[current_scale++].erase();
        internal_batch_nearest_neighbor(query, cover_sets, zero_set,
                                        current_scale, max_scale, upper_bound, results,
                                        spare_cover_sets, spare_zero_sets);
    }
}

template <class P> void batch_nearest_neighbor(const node<P> &top_node, const node<P> &query,
        v_array<v_array<P> > &results)
{
    v_array<v_array<v_array<d_node<P> > > > spare_cover_sets;
    v_array<v_array<d_node<P> > > spare_zero_sets;

    v_array<v_array<d_node<P> > > cover_sets = get_cover_sets(spare_cover_sets);
    v_array<d_node<P> > zero_set = pop(spare_zero_sets);

    float64_t* upper_bound = alloc_upper();
    setter(upper_bound,MAXFLOAT);

    float64_t top_dist = P::distance(query.p, top_node.p, MAXFLOAT);
    update(upper_bound, top_dist);

    d_node<P> temp = {top_dist, &top_node};
    push(cover_sets[0], temp);

    internal_batch_nearest_neighbor(&query,cover_sets,zero_set,0,0,upper_bound,results,
                                    spare_cover_sets,spare_zero_sets);

    SG_FREE(upper_bound);
    push(spare_cover_sets, cover_sets);

    for (int32_t i = 0; i < spare_cover_sets.index(); i++)
    {
        v_array<v_array<d_node<P> > > cover_sets = spare_cover_sets[i];
        for (int32_t j = 0; j < cover_sets.index(); j++)
            SG_FREE (cover_sets[j].begin);
        SG_FREE(cover_sets.begin);
    }
    SG_FREE(spare_cover_sets.begin);

    push(spare_zero_sets, zero_set);

    for (int32_t i = 0; i < spare_zero_sets.index(); i++)
        SG_FREE(spare_zero_sets[i].begin);
    SG_FREE(spare_zero_sets.begin);
}

template <class P> void k_nearest_neighbor(const node<P> &top_node, const node<P> &query,
        v_array<v_array<P> > &results, int32_t k)
{
    internal_k = k;
    update = update_k;
    setter = set_k;
    alloc_upper = alloc_k;

    batch_nearest_neighbor(top_node, query,results);
}

template <class P> void epsilon_nearest_neighbor(const node<P> &top_node, const node<P> &query,
        v_array<v_array<P> > &results, float64_t epsilon)
{
    internal_epsilon = epsilon;
    update = update_epsilon;
    setter = set_epsilon;
    alloc_upper = alloc_epsilon;

    batch_nearest_neighbor(top_node, query,results);
}

template <class P> void unequal_nearest_neighbor(const node<P> &top_node, const node<P> &query,
        v_array<v_array<P> > &results)
{
    update = update_unequal;
    setter = set_unequal;
    alloc_upper = alloc_unequal;

    batch_nearest_neighbor(top_node, query, results);
}

}
#endif


