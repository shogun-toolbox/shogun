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

#include <shogun/lib/JLCoverTreePoint.h>


namespace shogun { 

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

	SG_SERROR("Something has gone wrong, case not handled\n")
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

/** Print the information of the CoverTree point */
void print(CJLCoverTreePoint &p)
{
	SG_SERROR("Print JLCoverTreePoint not implemented\n")
}

}
