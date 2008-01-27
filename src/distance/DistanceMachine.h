/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 2006-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DISTANCE_MACHINE_H__
#define _DISTANCE_MACHINE_H__

#include "lib/common.h"
#include "distance/Distance.h"
#include "features/Labels.h"
#include "classifier/Classifier.h"

#include <stdio.h>

/** class DistanceMachine */
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
		inline void set_distance(CDistance* d) { distance=d; }

		/** get distance
		 *
		 * @return distance
		 */
		inline CDistance* get_distance() { return distance; }
		
	protected:
		/** the distance */
		CDistance* distance;
};
#endif
