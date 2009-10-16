/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2009 Christian Gehl
 * Written (W) 2006-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DISTANCE_MACHINE_H__
#define _DISTANCE_MACHINE_H__

#include "lib/common.h"
#include "distance/Distance.h"
#include "features/Labels.h"
#include "classifier/Classifier.h"

#include <stdio.h>

namespace shogun
{

/** @brief A generic DistanceMachine interface.
 *
 * A distance machine is based on a a-priori choosen distance.
 */
class CDistanceMachine : public CClassifier
{
	public:
		/** default constructor */
		CDistanceMachine();
		virtual ~CDistanceMachine();

		/** set distance
		 *
		 * @param d distance to set
		 */
		inline void set_distance(CDistance* d)
		{
			SG_UNREF(distance);
			SG_REF(d);
			distance=d;
		}

		/** get distance
		 *
		 * @return distance
		 */
		inline CDistance* get_distance() { SG_REF(distance); return distance; }
		
	protected:
		/** the distance */
		CDistance* distance;
};
}
#endif
