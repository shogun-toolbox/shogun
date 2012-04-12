/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) John Langford and Dinoj Surendran, v_array and its templatization
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _JLCTPOINT_H__
#define _JLCTPOINT_H__

#include <shogun/lib/config.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/Features.h>

namespace shogun
{

/* v_array class taken directly from JL's implementation */

template<class T> 
class v_array{

	public:
		T last() { return elements[index-1];}
		void decr() { index--;}
		v_array() { index = 0; length=0; elements = NULL;}
		T& operator[](unsigned int i) { return elements[i]; }

	public:
		int index;
		int length;
		T* elements;

};

template<class T> 
void push(v_array<T>& v, const T &new_ele)
{
	while(v.index >= v.length)
	{
		v.length = 2*v.length + 3;
		v.elements = (T *)realloc(v.elements,sizeof(T) * v.length);
	}
	v[v.index++] = new_ele;
}

template<class T> 
void alloc(v_array<T>& v, int length)
{
	v.elements = (T *)realloc(v.elements, sizeof(T) * length);
	v.length = length;
}
 
template<class T> 
v_array<T> pop(v_array<v_array<T> > &stack)
{
	if (stack.index > 0)
		return stack[--stack.index];
	else
		return v_array<T>();
}

enum EFeaturesContainer
{
	FC_LHS = 0,
	FC_RHS = 1,
};

class CJLCoverTreePoint
{

	public:

		/** Distance object where to find the coordinate information of 
		 * this point */
		CDistance* m_distance;

		/** Index of this point in m_distance */
		int32_t m_index;

		/** If the point is stored in rhs or lhs in m_distance */
		EFeaturesContainer m_features_container;

}; /* class JLCoverTreePoint */

/** Functions declared out of the class definition to respect JLCoverTree 
 *  structure */

float64_t distance(CJLCoverTreePoint p1, CJLCoverTreePoint p2, float64_t upper_bound);

/** Fills up a v_array of CJLCoverTreePoint objects */
v_array< CJLCoverTreePoint > parse_points(CDistance* distance, EFeaturesContainer fc);

void print(CJLCoverTreePoint &p);

} /* namespace shogun */

#endif /* _JLCTPOINT_H__*/
