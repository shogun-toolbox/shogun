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

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/Features.h>

using namespace distance;

namespace shogun
{
class CSGInterface;

/** @brief UI distance */
class CGUIDistance : public CSGObject
{
 public:
	/** constructor */
	CGUIDistance() {};
	/** constructor
	 * @param interface
	 */
	CGUIDistance(CSGInterface* interface);
	/** destructor */
	~CGUIDistance();

	/** get current distance */
	distance::CDistance* get_distance();
	/** set new distance */
	bool set_distance(CDistance* dist);

	/** create generic distance given by type */
	CDistance* create_generic(EDistanceType type);
	/** create Minkowski Metric */
	CDistance* create_minkowski(float64_t k=3);
	/** create HammingWord Distance */
	CDistance* create_hammingword(bool use_sign=false);

	/** initialize distance */
	bool init_distance(const char* target);
	/** save distance
	 * @param param
	 */
	bool save_distance(char* param);

	/** is initialized */
	bool is_initialized() { return initialized; }

	/** @return object name */
	virtual const char* get_name() const { return "GUIDistance"; }

 protected:
	/** distance */
	CDistance* distance;
	/** ui */
	CSGInterface* ui;
	/** initialized */
	bool initialized;
};
}
#endif //__GUIDISTANCE_H__
