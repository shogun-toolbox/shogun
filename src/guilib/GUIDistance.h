/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GUIDISTANCE_H__
#define __GUIDISTANCE_H__

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "distance/Distance.h"
#include "features/Features.h"

class CGUI ;

class CGUIDistance
{
 public:
	CGUIDistance(CGUI*);
	~CGUIDistance();

	CDistance* get_distance();
	bool set_distance(CHAR* param);
	CDistance* create_distance(CHAR* params);
	bool init_distance(CHAR* param);
	bool load_distance_init(CHAR* param);
	bool save_distance_init(CHAR* param);
	bool save_distance(CHAR* param);

	bool clean_distance(CHAR* param);

	bool is_initialized() { return initialized ; } ;

 protected:
	CDistance* distance;
	CGUI* gui ;
	bool initialized;
};
#endif //HAVE_SWIG
#endif
