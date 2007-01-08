/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 2006 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DISTANCE_MACHINE_H__
#define _DISTANCE_MACHINE_H__

#include "lib/common.h"
#include "distance/Distance.h"
#include "features/Labels.h"
#include "classifier/Classifier.h"

#include <stdio.h>

class CDistanceMachine : public CClassifier
{
	public:
		CDistanceMachine();
		virtual ~CDistanceMachine();

		inline void set_distance(CDistance* d) { distance=d; }
		inline CDistance* get_distance() { return distance; }
		
	protected:
		CDistance* distance;
};
#endif
