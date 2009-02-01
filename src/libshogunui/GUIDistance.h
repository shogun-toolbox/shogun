/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GUIDISTANCE_H__
#define __GUIDISTANCE_H__

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "base/SGObject.h"
#include "distance/Distance.h"
#include "features/Features.h"

class CSGInterface;

class CGUIDistance : public CSGObject
{
 public:
	CGUIDistance(CSGInterface* interface);
	~CGUIDistance();

	/** get current distance */
	CDistance* get_distance();
	/** set new distance */
	bool set_distance(CDistance* dist);

	/** create generic distance given by type */
	CDistance* create_generic(EDistanceType type);
	/** create Minkowski Metric */
	CDistance* create_minkowski(float64_t k=3);
	/** create HammingWord Distance */
	CDistance* create_hammingword(bool use_sign=false);

	/** initialize distance */
	bool init_distance(char* target);
	bool load_distance_init(char* param);
	bool save_distance_init(char* param);
	bool save_distance(char* param);

	bool is_initialized() { return initialized; }

	/** @return object name */
	inline virtual const char* get_name() const { return "GUIDistance"; }

 protected:
	CDistance* distance;
	CSGInterface* ui;
	bool initialized;
};
#endif //HAVE_SWIG
#endif //__GUIDISTANCE_H__
