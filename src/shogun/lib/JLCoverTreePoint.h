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

float distance(CJLCoverTreePoint p1, CJLCoverTreePoint p2, float64_t upper_bound)
{
	/** Call m_distance->distance() with the proper index order depending on 
	 *  the feature containers in m_distance for each of the points*/

	if ( p1.m_features_container == p2.m_features_container )
	{
		if ( ! p1.m_distance->lhs_equals_rhs() )
		{
			SG_SERROR("lhs != rhs but the distance of two points "
			       "from the same container has been requested\n");
		}
		else
		{
			return p1.m_distance->distance_upper_bounded(p1.m_index, 
					p2.m_index, upper_bound);
		}
	}
	else
	{
		if ( p1.m_distance->lhs_equals_rhs() )
		{
			SG_SERROR("lhs == rhs but the distance of two points "
			      "from different containers has been requested\n");
		}
		else
		{
			if ( p1.m_features_container == FC_LHS )
			{
				return p1.m_distance->distance_upper_bounded(p1.m_index, 
						p2.m_index, upper_bound);
			}
			else
			{
				return p1.m_distance->distance_upper_bounded(p2.m_index, 
						p1.m_index, upper_bound);
			}
		}
	}

	SG_SERROR("Something has gone wrong, case not handled\n");
	return -1;
}

/** Fills up a v_array of CJLCoverTreePoint objects */
v_array< CJLCoverTreePoint > parse_points(CDistance* distance, EFeaturesContainer fc)
{
	CFeatures* features;
	if ( fc == FC_LHS )
		features = distance->get_lhs();
	else
		features = distance->get_rhs();

	v_array< CJLCoverTreePoint > parsed;
	for ( int32_t i = 0 ; i < features->get_num_vectors() ; ++i )
	{
		CJLCoverTreePoint new_point;

		new_point.m_distance = distance;
		new_point.m_index = i;
		new_point.m_features_container = fc;

		push(parsed, new_point);
	}
	
	return parsed;
}

void print(CJLCoverTreePoint &p)
{
	SG_SERROR("Print JLCoverTreePoint not implemented\n");
}

} /* namespace shogun */

#endif /* _JLCTPOINT_H__*/
